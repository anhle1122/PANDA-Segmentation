"""Option 3: slide-bag UNI2+UPerNet with dual ISUP losses.

Per slide (one optimizer step):
  1. Micro-batch patches through seg head → CE+Dice on masks
  2. After the full bag: L_slide = derived-ISUP-from-seg vs clinician
  3. After the full bag: L_grade = grade head (pooled feats) vs clinician
  L = L_pixel + λ_slide * L_slide + λ_grade * L_grade
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from patch_utils import OUTPUTS  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402
from train.class_weights import get_or_compute_class_weights  # noqa: E402
from train.grade_head import (  # noqa: E402
    ISUPGradeHead,
    aggregate_softmax_probs,
    derived_isup_ce_from_seg_probs,
    grade_head_ce,
)
from train.losses import segmentation_loss  # noqa: E402
from train.metrics import PerClassDiceAccumulator  # noqa: E402
from train.slide_bag_dataset import (  # noqa: E402
    SlideBagPatchDataset,
    slide_bag_collate,
    summarize_bags,
)
from train.uni2_upernet import DEFAULT_FPN_CHANNELS, build_uni2_upernet  # noqa: E402
from train_baseline import (  # noqa: E402
    cleanup_distributed,
    is_main_process,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    setup_distributed,
    subsample_split_csv,
    unwrap_model,
)

SPLITS_DIR = OUTPUTS / "splits"


class SegPlusGrade(nn.Module):
    """Thin wrapper so DDP sees a single forward returning (logits, feats, grade_logits)."""

    def __init__(self, seg: nn.Module, grade_head: nn.Module) -> None:
        super().__init__()
        self.seg = seg
        self.grade_head = grade_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, feats = self.seg.forward_with_features(x)
        grade_logits = self.grade_head(feats)
        return logits, feats, grade_logits

    def backbone_parameters(self):
        return self.seg.backbone_parameters()

    def decoder_parameters(self):
        yield from self.seg.decoder_parameters()
        yield from self.grade_head.parameters()

    def freeze_backbone(self) -> None:
        self.seg.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        self.seg.unfreeze_backbone()


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, device = setup_distributed()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = OUTPUTS / "checkpoints" / f"uni2_upernet_{args.mode}_{args.run_tag}"
    if is_main_process(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.csv"

    train_ds = SlideBagPatchDataset(
        SPLITS_DIR / "panda_train.csv",
        max_patches_per_slide=args.max_patches_per_slide,
        mode=args.mode,
        allow_missing_h5=args.allow_missing_h5,
        seed=args.seed,
    )
    val_csv = SPLITS_DIR / "panda_val.csv"
    if args.max_val_patches:
        val_csv = subsample_split_csv(val_csv, args.max_val_patches, args.seed)
    val_ds = BaselinePatchDataset(
        val_csv, mode=args.mode, allow_missing_h5=args.allow_missing_h5
    )

    if is_main_process(rank):
        print("Option 3 slide-bag |", summarize_bags(train_ds))
        print(
            f"λ_slide={args.lambda_slide} λ_grade={args.lambda_grade} "
            f"micro_bs={args.micro_batch_size} slides/ep={args.slides_per_epoch}"
        )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=slide_bag_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    weight_df = pd.read_csv(SPLITS_DIR / "panda_train.csv")
    weight_bundle = get_or_compute_class_weights(weight_df, mode=args.mode)
    class_weights = weight_bundle["class_weights"].to(device)

    seg = build_uni2_upernet(
        num_classes=6,
        pretrained=True,
        freeze_backbone=args.freeze_backbone_epochs > 0,
        checkpoint_path=args.uni2_checkpoint or None,
    )
    grade_head = ISUPGradeHead(DEFAULT_FPN_CHANNELS[-1], num_isup=6)
    model: nn.Module = SegPlusGrade(seg, grade_head).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    def _build_optim(lr: float, backbone_on: bool) -> torch.optim.Optimizer:
        core = unwrap_model(model)
        groups = [
            {
                "params": [p for p in core.seg.decoder_parameters() if p.requires_grad],
                "lr": lr,
            },
            {"params": list(core.grade_head.parameters()), "lr": lr},
        ]
        if backbone_on:
            bb = [p for p in core.seg.backbone_parameters() if p.requires_grad]
            if bb:
                groups.append({"params": bb, "lr": lr * args.backbone_lr_mult})
        return torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)

    backbone_frozen = args.freeze_backbone_epochs > 0
    optimizer = _build_optim(args.lr, backbone_on=not backbone_frozen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_cancer = -1.0
    if args.resume and Path(args.resume).is_file():
        start_epoch = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        if is_main_process(rank):
            print(f"Resumed from {args.resume} → next epoch {start_epoch}")

    if is_main_process(rank) and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "cancer_dice",
                    "mean_dice",
                    "L_pixel",
                    "L_slide",
                    "L_grade",
                    "lr",
                ]
            )

    micro = max(1, int(args.micro_batch_size))

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if backbone_frozen and epoch > args.freeze_backbone_epochs:
            unwrap_model(model).unfreeze_backbone()
            backbone_frozen = False
            optimizer = _build_optim(args.lr, backbone_on=True)
            if is_main_process(rank):
                print(f"=== Epoch {epoch}: unfroze backbone ===")

        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "pixel": 0.0, "slide": 0.0, "grade": 0.0, "n": 0}
        n_slides = 0

        for bag in train_loader:
            if args.slides_per_epoch and n_slides >= args.slides_per_epoch:
                break

            images = bag["images"].to(device, non_blocking=True)
            masks = bag["masks"].to(device, non_blocking=True)
            weights = bag["weights"].to(device, non_blocking=True)
            isup = int(bag["isup"].item())
            n_patches = int(images.shape[0])

            optimizer.zero_grad(set_to_none=True)
            logits_chunks: list[torch.Tensor] = []
            feat_chunks: list[torch.Tensor] = []
            pixel_loss_acc = 0.0

            n_micro = int(math.ceil(n_patches / micro))
            for m_i in range(n_micro):
                sl = slice(m_i * micro, min((m_i + 1) * micro, n_patches))
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, feats, _grade = model(images[sl])
                    p_loss = segmentation_loss(
                        logits,
                        masks[sl],
                        weights[sl],
                        class_weights,
                        adjacent_soft_alpha=args.adjacent_soft_alpha,
                    )
                    scaled = p_loss * ((sl.stop - sl.start) / n_patches)
                scaler.scale(scaled).backward()
                pixel_loss_acc += float(scaled.detach().item())
                logits_chunks.append(logits.detach())
                feat_chunks.append(feats.detach())

            # Differentiable slide losses on a random subset (+ bag-mean prior)
            idx = torch.randperm(n_patches, device=device)[: min(micro, n_patches)]
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_g, feats_g, grade_live = model(images[idx])
                mean_live = torch.softmax(logits_g.float(), dim=1).mean(dim=(0, 2, 3))
                bag_mean = aggregate_softmax_probs(logits_chunks).to(mean_live.device)
                mean_probs = 0.5 * mean_live + 0.5 * bag_mean
                l_slide, _ = derived_isup_ce_from_seg_probs(mean_probs, isup)

                feat_bag = torch.cat(feat_chunks, dim=0).mean(dim=0)
                feat_mix = 0.5 * feats_g.mean(dim=0) + 0.5 * feat_bag
                # Prefer live grade_logits from the subset forward for DDP sync
                g_logits = 0.5 * grade_live.mean(dim=0) + 0.5 * unwrap_model(model).grade_head(
                    feat_mix
                )
                l_grade = grade_head_ce(g_logits, isup)
                slide_term = args.lambda_slide * l_slide + args.lambda_grade * l_grade

            scaler.scale(slide_term).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total = pixel_loss_acc + float(slide_term.detach().item())
            running["loss"] += total
            running["pixel"] += pixel_loss_acc
            running["slide"] += float(l_slide.detach().item())
            running["grade"] += float(l_grade.detach().item())
            running["n"] += 1
            n_slides += 1

        # Validation — patch-wise Dice on original masks
        model.eval()
        dice_acc = PerClassDiceAccumulator(num_classes=6)  # keyword-only ok
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for images_v, masks_v, weights_v in val_loader:
                images_v = images_v.to(device, non_blocking=True)
                masks_v = masks_v.to(device, non_blocking=True)
                weights_v = weights_v.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits_v, _, _ = model(images_v)
                    vloss = segmentation_loss(
                        logits_v,
                        masks_v,
                        weights_v,
                        class_weights,
                        adjacent_soft_alpha=args.adjacent_soft_alpha,
                    )
                val_loss_sum += float(vloss.item())
                val_n += 1
                dice_acc.update(logits_v.argmax(1), masks_v)

        metrics = dice_acc.to_baseline_metrics()
        cancer = float(metrics.get("cancer_dice", 0.0))
        mean_dice = float(metrics.get("mean_dice", 0.0))
        train_loss = running["loss"] / max(1, running["n"])
        val_loss = val_loss_sum / max(1, val_n)
        scheduler.step()

        if is_main_process(rank):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.4f} "
                f"(pix={running['pixel']/max(1,running['n']):.4f} "
                f"slide={running['slide']/max(1,running['n']):.4f} "
                f"grade={running['grade']/max(1,running['n']):.4f}) "
                f"| val={val_loss:.4f} cancer={cancer:.4f} mean={mean_dice:.4f} "
                f"| {time.time()-t0:.0f}s"
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{cancer:.6f}",
                        f"{mean_dice:.6f}",
                        f"{running['pixel']/max(1,running['n']):.6f}",
                        f"{running['slide']/max(1,running['n']):.6f}",
                        f"{running['grade']/max(1,running['n']):.6f}",
                        lr,
                    ]
                )
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
            if cancer > best_cancer:
                best_cancer = cancer
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics=metrics,
                    class_weights=class_weights,
                    mode=args.mode,
                )
                print(f"  new best cancer_dice={best_cancer:.4f}")
            if epoch % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}_cancer_{cancer:.4f}.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics=metrics,
                    class_weights=class_weights,
                    mode=args.mode,
                )
                prune_checkpoints(ckpt_dir, keep=args.keep_checkpoints)

        if world_size > 1:
            dist.barrier()

    if is_main_process(rank):
        (ckpt_dir / "TRAINING_COMPLETE.txt").write_text(
            f"best_cancer={best_cancer}\n", encoding="utf-8"
        )
        print(f"DONE — best val cancer_dice={best_cancer:.4f}")
    cleanup_distributed()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Option 3 slide-bag + dual ISUP losses")
    p.add_argument("--mode", default="raw", choices=["raw", "normalized", "normalized_ink_raw"])
    p.add_argument("--run-tag", default="pseudo_r1_opt3_slidebag")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--backbone-lr-mult", type=float, default=0.05)
    p.add_argument("--freeze-backbone-epochs", type=int, default=5)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--slides-per-epoch", type=int, default=256)
    p.add_argument("--max-patches-per-slide", type=int, default=None)
    p.add_argument("--max-val-patches", type=int, default=20000)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lambda-slide", type=float, default=0.3)
    p.add_argument("--lambda-grade", type=float, default=0.3)
    p.add_argument("--adjacent-soft-alpha", type=float, default=0.22)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--allow-missing-h5", action="store_true")
    p.add_argument("--uni2-checkpoint", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.uni2_checkpoint:
        default_ckpt = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "ckpts"
            / "uni2-h"
            / "pytorch_model.bin"
        )
        args.uni2_checkpoint = str(default_ckpt) if default_ckpt.is_file() else ""
    train(args)
