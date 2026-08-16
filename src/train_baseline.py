"""Baseline Gleason training — PyTorch loop, DDP + AMP, three image modes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import OUTPUTS, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402
from train.class_weights import format_weight_table, get_or_compute_class_weights  # noqa: E402
from train.losses import segmentation_loss  # noqa: E402
from train.metrics import compute_per_class_dice  # noqa: E402
from train.model import build_model  # noqa: E402

SPLITS_DIR = OUTPUTS / "splits"
MODES = ("raw", "normalized", "normalized_ink_raw")


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return local_rank, rank, world_size, device
    return 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def subsample_split_csv(split_csv: Path, max_patches: int, seed: int) -> Path:
    df = pd.read_csv(split_csv)
    if len(df) <= max_patches:
        return split_csv
    tmp = SPLITS_DIR / f"_tmp_{split_csv.stem}_{max_patches}.csv"
    df.sample(n=max_patches, random_state=seed).to_csv(tmp, index=False)
    return tmp


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    metrics: dict,
    class_weights: torch.Tensor,
    mode: str,
    extra: dict | None = None,
) -> None:
    """Atomic checkpoint write. Never clobber an existing ``epoch_*.pth``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.startswith("epoch_") and path.exists():
        print(f"  keep existing {path.name} (will not overwrite)", flush=True)
        return
    payload = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_mean_dice": metrics["mean_dice"],
        "val_cancer_dice": metrics["cancer_dice"],
        "class_weights": class_weights.cpu(),
        "mode": mode,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def restore_best_cancer_dice(ckpt_dir: Path) -> float:
    """Best val cancer Dice from the log / named snapshots — never from latest.pth alone.

    Resume used to reset best to -1, so a worse epoch overwrote ``best.pth``
    (Omar-6 ep7 0.608 → ep16 0.521 on 2026-08-15).
    """
    best = -1.0
    log_path = Path(ckpt_dir) / "training_log.csv"
    if log_path.is_file():
        try:
            df = pd.read_csv(log_path)
            if "cancer_dice" in df.columns and len(df):
                best = max(best, float(df["cancer_dice"].max()))
        except Exception as exc:
            print(f"  warn: could not read {log_path}: {exc}", flush=True)
    for p in Path(ckpt_dir).glob("epoch_*_cancer_*.pth"):
        try:
            best = max(best, float(p.stem.split("cancer_")[-1]))
        except ValueError:
            continue
    return best


def epoch_snapshot_path(ckpt_dir: Path, epoch: int, cancer: float) -> Path:
    return Path(ckpt_dir) / f"epoch_{int(epoch):03d}_cancer_{float(cancer):.4f}.pth"


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = int(ckpt["epoch"]) + 1
    return start_epoch


def prune_checkpoints(ckpt_dir: Path, keep: int = 0) -> None:
    """Optionally drop old ``epoch_*.pth`` files.

    ``keep <= 0`` means keep **all** epoch checkpoints (default). This avoids
    silently deleting externally useful epochs (e.g. Opt3 ep12 pruned while
    later worse-val epochs were retained by mtime). ``best.pth`` / ``latest.pth``
    are never touched. Pass a positive ``keep`` only when disk pressure forces it.
    """
    if keep is None or int(keep) <= 0:
        return
    # 2026-08-15: never auto-delete named epoch snapshots (ep7 / ep12 losses).
    print(
        f"  prune_checkpoints(keep={keep}) ignored — epoch_*.pth are immutable",
        flush=True,
    )
    return


def make_train_loader(
    dataset: BaselinePatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    patches_per_epoch: int | None,
    pin_memory: bool,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    n = len(dataset)
    if patches_per_epoch and patches_per_epoch < n:
        per_rank = max(1, patches_per_epoch // world_size)
        g = torch.Generator()
        g.manual_seed(seed + epoch * 1000 + rank)
        sampler = RandomSampler(
            dataset,
            num_samples=per_rank,
            replacement=False,
            generator=g,
        )
        return DataLoader(dataset, sampler=sampler, **loader_kwargs)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        sampler.set_epoch(epoch)
        return DataLoader(dataset, sampler=sampler, **loader_kwargs)

    return DataLoader(dataset, shuffle=True, **loader_kwargs)


def log_epoch_csv(log_path: Path, row: dict, *, write_header: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, device = setup_distributed()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = OUTPUTS / "checkpoints" / f"baseline_{args.mode}"
    if is_main_process(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.csv"

    train_csv = SPLITS_DIR / "panda_train.csv"
    val_csv = SPLITS_DIR / "panda_val.csv"
    if args.max_patches:
        train_csv = subsample_split_csv(train_csv, args.max_patches, args.seed)
        val_csv = subsample_split_csv(val_csv, max(20, args.max_patches // 5), args.seed)

    train_ds = BaselinePatchDataset(
        train_csv, mode=args.mode, allow_missing_h5=args.allow_missing_h5,
    )
    val_ds = BaselinePatchDataset(
        val_csv, mode=args.mode, allow_missing_h5=args.allow_missing_h5,
    )
    if is_main_process(rank):
        train_ds.run_sanity_check(n=5)

    val_loader_kwargs: dict = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(val_ds, **val_loader_kwargs) if is_main_process(rank) else None

    train_df = pd.read_csv(train_csv)
    weight_df = (
        pd.read_csv(SPLITS_DIR / "panda_train.csv")
        if args.max_patches
        else train_df
    )
    weight_bundle = get_or_compute_class_weights(
        weight_df,
        mode=args.mode,
        recompute=args.recompute_freq,
    )
    class_weights = weight_bundle["class_weights"].to(device)

    model = build_model(num_classes=6).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer, scheduler, scaler)
        if is_main_process(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if is_main_process(rank):
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"World size:    {world_size}")
        print(f"AMP:           {use_amp}")
        print(f"Mode:          {args.mode}")
        train_per_epoch = (
            min(args.patches_per_epoch, len(train_ds))
            if args.patches_per_epoch
            else len(train_ds)
        )
        if world_size > 1 and args.patches_per_epoch:
            train_per_epoch = (train_per_epoch // world_size) * world_size
        print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")
        print(f"Patches/epoch: {train_per_epoch} (total across {world_size} GPU(s))")
        print(format_weight_table(weight_bundle))
        print(f"Checkpoint dir: {ckpt_dir}")
        print(f"Log path:       {log_path}")
        print(f"Epochs:         {start_epoch}..{args.epochs}")

    best_val_dice = 0.0
    patience_counter = 0
    write_header = not log_path.exists() or args.resume is None

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loader = make_train_loader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patches_per_epoch=args.patches_per_epoch,
            pin_memory=device.type == "cuda",
            seed=args.seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        model.train()
        train_losses = []
        for images, masks, weights in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = segmentation_loss(logits, masks, weights, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))

        metrics = {"mean_dice": 0.0, "cancer_dice": 0.0}
        val_loss = 0.0
        if is_main_process(rank):
            model.eval()
            val_losses = []
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, masks, weights in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    weights = weights.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(images)
                        loss = segmentation_loss(logits, masks, weights, class_weights)
                    val_losses.append(loss.item())
                    all_preds.append(logits.float().cpu())
                    all_targets.append(masks.cpu())
            val_loss = float(np.mean(val_losses))
            metrics = compute_per_class_dice(torch.cat(all_preds), torch.cat(all_targets))

        if world_size > 1:
            dist.barrier()

        if is_main_process(rank):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s"
            )
            print(
                f"  background: {metrics['dice_0']:.3f}  stroma: {metrics['dice_1']:.3f}  "
                f"benign: {metrics['dice_2']:.3f}"
            )
            print(
                f"  G3: {metrics['dice_3']:.3f}  G4: {metrics['dice_4']:.3f}  "
                f"G5: {metrics['dice_5']:.3f}"
            )
            print(
                f"  mean_dice: {metrics['mean_dice']:.3f}  cancer_dice: {metrics['cancer_dice']:.3f}"
            )

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "dice_bg": round(metrics["dice_0"], 4),
                "dice_stroma": round(metrics["dice_1"], 4),
                "dice_benign": round(metrics["dice_2"], 4),
                "dice_g3": round(metrics["dice_3"], 4),
                "dice_g4": round(metrics["dice_4"], 4),
                "dice_g5": round(metrics["dice_5"], 4),
                "mean_dice": round(metrics["mean_dice"], 4),
                "cancer_dice": round(metrics["cancer_dice"], 4),
                "lr": lr,
            }
            log_epoch_csv(log_path, row, write_header=write_header)
            write_header = False

            if metrics["mean_dice"] > best_val_dice:
                best_val_dice = metrics["mean_dice"]
                patience_counter = 0
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}_dice_{metrics['mean_dice']:.4f}.pth"
                save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics=metrics,
                    class_weights=class_weights,
                    mode=args.mode,
                )
                shutil.copy2(ckpt_path, ckpt_dir / "best.pth")
                prune_checkpoints(ckpt_dir, keep=0)
            else:
                patience_counter += 1
                if epoch >= args.min_epochs and patience_counter >= args.patience:
                    print(
                        f"Early stopping at epoch {epoch} — "
                        f"no val mean_dice improvement for {args.patience} epochs"
                    )
                    break

    if is_main_process(rank):
        done_path = ckpt_dir / "TRAINING_COMPLETE.txt"
        done_path.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "epochs_run": epoch,
                    "best_val_mean_dice": best_val_dice,
                    "checkpoint_dir": str(ckpt_dir),
                    "log": str(log_path),
                    "world_size": world_size,
                    "amp": use_amp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nDONE — best val mean_dice={best_val_dice:.4f}")
        print(f"Completion marker: {done_path}")

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patches-per-epoch", type=int, default=50_000)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--allow-missing-h5", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--recompute-freq", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.patches_per_epoch == 0:
        args.patches_per_epoch = None
    train(args)


if __name__ == "__main__":
    main()
