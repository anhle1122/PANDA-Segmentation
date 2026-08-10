"""UNI2-h + UPerNet Gleason training — same data/loss as Model A (no focal / no LS)."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import OUTPUTS  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402
from train.class_weights import format_weight_table, get_or_compute_class_weights  # noqa: E402
from train.losses import isup_informed_segmentation_loss, segmentation_loss  # noqa: E402
from train.pseudo_label_dataset import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PRED_DIR,
    PseudoLabelPatchDataset,
    build_corrected_target,
)
from train.metrics import PerClassDiceAccumulator  # noqa: E402
from train.uni2_upernet import build_uni2_upernet  # noqa: E402
from train_baseline import (  # noqa: E402
    cleanup_distributed,
    is_main_process,
    load_checkpoint,
    log_epoch_csv,
    make_train_loader,
    prune_checkpoints,
    save_checkpoint,
    setup_distributed,
    subsample_split_csv,
    unwrap_model,
)

SPLITS_DIR = OUTPUTS / "splits"
MODES = ("raw", "normalized", "normalized_ink_raw")


def _build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
    backbone_trainable: bool,
) -> torch.optim.Optimizer:
    core = unwrap_model(model)
    decoder_params = [p for p in core.decoder_parameters() if p.requires_grad]
    param_groups = [{"params": decoder_params, "lr": lr}]
    if backbone_trainable:
        backbone_params = [p for p in core.backbone_parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, device = setup_distributed()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = OUTPUTS / "checkpoints" / (
        f"uni2_upernet_{args.mode}_{args.run_tag}"
        if args.run_tag
        else f"uni2_upernet_{args.mode}"
    )
    if is_main_process(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.csv"

    train_csv = SPLITS_DIR / "panda_train.csv"
    val_csv = SPLITS_DIR / "panda_val.csv"
    if args.max_patches:
        train_csv = subsample_split_csv(train_csv, args.max_patches, args.seed)
        val_csv = subsample_split_csv(val_csv, max(20, args.max_patches // 5), args.seed)
    elif args.max_val_patches:
        # Deterministic (seeded) subsample, so the val set is identical across
        # epochs, ranks and rounds -- otherwise checkpoint selection and the
        # round-over-round comparison would be measuring different things.
        val_csv = subsample_split_csv(val_csv, args.max_val_patches, args.seed)

    # Pseudo-label mode swaps the train dataset (adds per-pixel corrections and
    # the round's seg_target) and the train loss. Validation deliberately stays
    # on the ORIGINAL mask with the plain segmentation loss in every round, so
    # val metrics remain comparable across rounds.
    use_pseudo = args.pseudo_label
    if use_pseudo:
        train_ds = PseudoLabelPatchDataset(
            train_csv,
            manifest_csv=args.pseudo_manifest,
            pred_dir=args.pseudo_pred_dir,
            seg_target_dir=args.seg_target_dir,
            allow_missing_cache=args.allow_missing_cache,
            mode=args.mode,
            allow_missing_h5=args.allow_missing_h5,
            augment=bool(args.augment),
            augment_mode="patch",
        )
    else:
        train_ds = BaselinePatchDataset(
            train_csv,
            mode=args.mode,
            allow_missing_h5=args.allow_missing_h5,
            augment=bool(args.augment),
            augment_mode="patch",
        )
    val_ds = BaselinePatchDataset(
        val_csv,
        mode=args.mode,
        allow_missing_h5=args.allow_missing_h5,
        augment=False,
    )
    if is_main_process(rank):
        (train_ds.base if use_pseudo else train_ds).run_sanity_check(n=5)
        if use_pseudo:
            print(f"Pseudo-label mode ON | seg_target={train_ds.seg_target_mode}")
            print(f"  manifest:  {args.pseudo_manifest}")
            print(f"  pred cache: {args.pseudo_pred_dir}")
            print(
                "  loss: ISUP-informed segmentation "
                "(Rules 1-3 rewrite flagged pixels in the seg target; no fighting dual loss)"
            )
            counts = train_ds.rule_counts()
            for rule, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {rule:26s} {count:7d} patches")

    val_workers = 0 if world_size > 1 else args.num_workers
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    val_loader_kwargs: dict = {
        "batch_size": args.val_batch_size,
        "shuffle": val_sampler is None,
        "sampler": val_sampler,
        "num_workers": val_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        # Batch-stats BN (no running mean) cannot forward a size-1 batch through
        # the PPM's 1x1 pooled branch; drop a partial last batch rather than crash.
        "drop_last": True,
    }
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

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
        rank=rank,
        world_size=world_size,
    )
    class_weights = weight_bundle["class_weights"].to(device)

    freeze_now = args.freeze_backbone_epochs > 0
    model = build_uni2_upernet(
        num_classes=6,
        freeze_backbone=freeze_now,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.uni2_checkpoint,
        model_size=args.model_size,
    ).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = _build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult,
        backbone_trainable=not freeze_now,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer, scheduler, scaler)
        # Restore freeze state consistent with resumed epoch.
        core = unwrap_model(model)
        if start_epoch <= args.freeze_backbone_epochs:
            core.freeze_backbone()
        else:
            core.unfreeze_backbone()
            optimizer = _build_optimizer(
                model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                backbone_lr_mult=args.backbone_lr_mult,
                backbone_trainable=True,
            )
            # Keep scheduler progress from checkpoint when possible.
            try:
                scheduler.load_state_dict(
                    torch.load(args.resume, map_location="cpu", weights_only=False)[
                        "scheduler_state_dict"
                    ]
                )
            except Exception:  # noqa: BLE001
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, last_epoch=start_epoch - 1
                )
        if is_main_process(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if is_main_process(rank):
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            print(
                f"GPU memory: "
                f"{torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB"
            )
        print(f"World size:    {world_size}")
        print(f"AMP:           {use_amp}")
        print(f"Mode:          {args.mode}")
        print("Loss:          0.5 weighted CE + 0.5 soft Dice (no focal, no label smoothing)")
        print(f"Freeze epochs: {args.freeze_backbone_epochs}")
        print(f"Backbone LR×:  {args.backbone_lr_mult}")
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
        print(f"Select best by: cancer_dice")
        print(f"Epochs:         {start_epoch}..{args.epochs}")

    best_cancer_dice = -1.0
    patience_counter = 0
    write_header = not log_path.exists() or args.resume is None
    persist_workers = args.persistent_workers and args.num_workers > 0
    backbone_unfrozen = not freeze_now

    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: freeze UNI2 → train head → low-LR unfreeze.
        if (not backbone_unfrozen) and epoch > args.freeze_backbone_epochs:
            unwrap_model(model).unfreeze_backbone()
            optimizer = _build_optimizer(
                model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                backbone_lr_mult=args.backbone_lr_mult,
                backbone_trainable=True,
            )
            remaining = max(1, args.epochs - epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            backbone_unfrozen = True
            if is_main_process(rank):
                print(
                    f"=== Epoch {epoch}: unfroze UNI2-h backbone "
                    f"(lr_backbone={args.lr * args.backbone_lr_mult:.2e}) ==="
                )

        t0 = time.time()
        train_loader = make_train_loader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patches_per_epoch=args.patches_per_epoch,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
            persistent_workers=persist_workers,
            seed=args.seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            drop_last=True,
        )
        model.train()
        train_losses = []
        seg_losses: list[float] = []
        pixels_rewritten = 0
        for batch in train_loader:
            if use_pseudo:
                images, masks, weights, flag_masks, target_params = batch
                flag_masks = flag_masks.to(device, non_blocking=True)
                target_params = target_params.to(device, non_blocking=True)
            else:
                images, masks, weights = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                if use_pseudo:
                    corrected_target = build_corrected_target(flag_masks, target_params)
                    loss, loss_parts = isup_informed_segmentation_loss(
                        logits,
                        masks,
                        corrected_target,
                        flag_masks,
                        class_weights,
                        weights,
                        adjacent_soft_alpha=args.adjacent_soft_alpha,
                        g45_soft_alpha=args.g45_soft_alpha,
                        label_smoothing=args.label_smoothing,
                    )
                    seg_losses.append(loss_parts["seg"])
                    pixels_rewritten += loss_parts["pseudo_pixels_flagged"]
                else:
                    loss = segmentation_loss(
                        logits,
                        masks,
                        weights,
                        class_weights,
                        label_smoothing=args.label_smoothing,
                        adjacent_soft_alpha=args.adjacent_soft_alpha,
                        g45_soft_alpha=args.g45_soft_alpha,
                    )
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))

        eval_model = unwrap_model(model)
        eval_model.eval()
        dice_acc = PerClassDiceAccumulator(num_classes=6)
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for images, masks, weights in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = eval_model(images)
                    loss = segmentation_loss(
                        logits,
                        masks,
                        weights,
                        class_weights,
                        label_smoothing=args.label_smoothing,
                        adjacent_soft_alpha=args.adjacent_soft_alpha,
                        g45_soft_alpha=args.g45_soft_alpha,
                    )
                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                dice_acc.update(logits.argmax(dim=1), masks)

        if world_size > 1:
            loss_stats = torch.tensor(
                [val_loss_sum, float(val_n)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            val_loss_sum, val_n = loss_stats[0].item(), int(loss_stats[1].item())
            inter = dice_acc.intersection.to(device=device, dtype=torch.float64)
            union = dice_acc.union.to(device=device, dtype=torch.float64)
            dist.all_reduce(inter, op=dist.ReduceOp.SUM)
            dist.all_reduce(union, op=dist.ReduceOp.SUM)
            dice_acc.intersection = inter.cpu()
            dice_acc.union = union.cpu()

        val_loss = val_loss_sum / max(val_n, 1)
        metrics = dice_acc.to_baseline_metrics()

        # All ranks see the same reduced val_loss; stop together so DDP does not hang.
        if not math.isfinite(val_loss):
            if is_main_process(rank):
                print(
                    f"ERROR: non-finite val_loss={val_loss} at epoch {epoch} — "
                    "stopping before writing a corrupted best checkpoint."
                )
            break

        if is_main_process(rank):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | frozen={not backbone_unfrozen} | {elapsed:.0f}s"
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
            if use_pseudo and seg_losses:
                print(
                    f"  train_seg_loss={np.mean(seg_losses):.4f}  "
                    f"pixels_rewritten_by_isup={pixels_rewritten}"
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
                "backbone_frozen": int(not backbone_unfrozen),
            }
            if use_pseudo:
                row["seg_loss"] = round(float(np.mean(seg_losses)), 6) if seg_losses else 0.0
                row["pixels_rewritten"] = pixels_rewritten
            log_epoch_csv(log_path, row, write_header=write_header)
            write_header = False

            if metrics["cancer_dice"] > best_cancer_dice:
                best_cancer_dice = metrics["cancer_dice"]
                patience_counter = 0
                ckpt_path = (
                    ckpt_dir
                    / f"epoch_{epoch:03d}_cancer_{metrics['cancer_dice']:.4f}.pth"
                )
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
                        f"no val cancer_dice improvement for {args.patience} epochs"
                    )
                    break

            save_checkpoint(
                ckpt_dir / "latest.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics=metrics,
                class_weights=class_weights,
                mode=args.mode,
            )

    if is_main_process(rank):
        done_path = ckpt_dir / "TRAINING_COMPLETE.txt"
        payload = {
            "arch": "uni2_h_upernet",
            "mode": args.mode,
            "epochs_run": epoch,
            "best_val_cancer_dice": best_cancer_dice,
            "checkpoint_dir": str(ckpt_dir),
            "log": str(log_path),
            "world_size": world_size,
            "amp": use_amp,
            "loss": "0.5 weighted CE + 0.5 soft Dice",
            "focal": False,
            "label_smoothing": args.label_smoothing,
            "adjacent_soft_alpha": args.adjacent_soft_alpha,
            "g45_soft_alpha": args.g45_soft_alpha,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
        }
        done_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDONE — best val cancer_dice={best_cancer_dice:.4f}")
        print(f"Completion marker: {done_path}")

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNI2-h + UPerNet on PANDA Gleason patches")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--patches-per-epoch", type=int, default=50_000)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--allow-missing-h5", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--recompute-freq", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-size", type=int, default=518)
    parser.add_argument(
        "--uni2-checkpoint",
        type=Path,
        default=None,
        help="Local UNI2-h pytorch_model.bin (skips HF download)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random UNI2-h init (debug only)",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--adjacent-soft-alpha", type=float, default=0.15)
    parser.add_argument(
        "--g45-soft-alpha",
        type=float,
        default=None,
        help="Optional stronger G4↔G5 soft mass (e.g. 0.22); None = use adjacent only",
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-val-patches", type=int, default=20_000)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)

    pseudo = parser.add_argument_group("iterative pseudo-label self-training")
    pseudo.add_argument(
        "--pseudo-label",
        action="store_true",
        help=(
            "ISUP-informed segmentation: Rules 1-3 rewrite flagged pixels in the "
            "training seg target (single loss; no dual-term fight)"
        ),
    )
    pseudo.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_MANIFEST)
    pseudo.add_argument("--pseudo-pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    pseudo.add_argument(
        "--seg-target-dir",
        type=Path,
        default=None,
        help=(
            "Round 2+: directory of cached previous-round predictions to use as "
            "the base seg target (before ISUP rewrites). Omit for Round 1 to use "
            "the original PANDA mask as the base."
        ),
    )
    pseudo.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Skip corrections for slides with no cached prediction instead of failing",
    )
    # Kept for CLI compatibility with older scripts; ignored by the new loss path.
    pseudo.add_argument("--w-seg", type=float, default=0.70)
    pseudo.add_argument("--w-pseudo", type=float, default=0.30)
    args = parser.parse_args()
    if args.patches_per_epoch == 0:
        args.patches_per_epoch = None
    train(args)


if __name__ == "__main__":
    main()
