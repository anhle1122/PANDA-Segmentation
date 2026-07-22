"""Train PANDA Gleason segmentation with per-sample loss tracking and val Dice.

Tracks per epoch:
  - val_dice_mean (foreground classes 1-5; NOT SDR)
  - per-patch train loss distribution (histogram PNG each N epochs)
  - per-slide mean_loss saved each epoch -> outputs/loss_tracking/epoch_NNN_per_slide.csv
  - After training, pick epoch via src/suggest_noise_epoch.py (not a fixed number)

Training data is filtered by QC only (src/clean_dataset.py). The 6th-place
noise_ratio_10 list is never applied here.

After a sufficiently converged checkpoint, run diagnostic overlap (no auto-drops):
  python src/check_noise_overlap.py --top-pct 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import OUTPUTS, PATCH_INDEX_CSV, SELECTED_JSON  # noqa: E402
from train.callbacks import PerSampleLossCallback  # noqa: E402
from train.dataset import (  # noqa: E402
    PandaPatchDataset,
    load_patch_index,
    split_patch_index,
)
from train.module import GleasonSegmentationModule  # noqa: E402

PROJECT = SRC.parent
DATA = PROJECT / "data"
CLEAN_CSV = DATA / "radboud_clean.csv"
LOSS_TRACKING_DIR = OUTPUTS / "loss_tracking"
HIGH_LOSS_CSV = OUTPUTS / "high_loss_slides.csv"


def build_dataloaders(
    *,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    selected_only: bool,
    clean_slides_csv: Path,
    patch_index_csv: Path,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    if not patch_index_csv.exists():
        raise FileNotFoundError(
            f"Missing {patch_index_csv}. Run: python src/extract_patches.py first."
        )

    patch_df = load_patch_index(patch_index_csv, clean_slides_csv=clean_slides_csv)

    if selected_only:
        if not SELECTED_JSON.exists():
            raise FileNotFoundError(f"Missing {SELECTED_JSON}")
        selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8")).get("selected", [])
        patch_df = patch_df[patch_df["image_id"].astype(str).isin(selected)].copy()

    train_df, val_df = split_patch_index(patch_df, val_fraction=val_fraction)
    train_ds = PandaPatchDataset(train_df, augment=True)
    val_ds = PandaPatchDataset(val_df, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PANDA Gleason segmentation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--loss-hist-every", type=int, default=1, help="Plot per-patch loss histogram every N epochs")
    parser.add_argument("--selected-only", action="store_true", help="Train on outputs/selected_slides.json only")
    parser.add_argument("--patch-index", type=Path, default=PATCH_INDEX_CSV)
    parser.add_argument("--clean-slides", type=Path, default=CLEAN_CSV)
    parser.add_argument("--max-epochs-checkpoint", action="store_true", help="Save best checkpoint by val_dice_mean")
    args = parser.parse_args()

    train_loader, val_loader, train_df, val_df = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        selected_only=args.selected_only,
        clean_slides_csv=args.clean_slides,
        patch_index_csv=args.patch_index,
    )

    print(f"Train patches: {len(train_df)} ({train_df['image_id'].nunique()} slides)")
    print(f"Val patches:   {len(val_df)} ({val_df['image_id'].nunique()} slides)")
    print("Val metric:    val_dice_mean (foreground classes 1-5)")

    module = GleasonSegmentationModule(lr=args.lr)

    callbacks: list = [
        PerSampleLossCallback(
            train_loader=train_loader,
            output_dir=LOSS_TRACKING_DIR,
            every_n_epochs=args.loss_hist_every,
            high_loss_csv=None,
        )
    ]
    if args.max_epochs_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=OUTPUTS / "checkpoints",
                filename="gleason-{epoch:02d}-{val_dice_mean:.4f}",
                monitor="val_dice_mean",
                mode="max",
                save_top_k=1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=args.max_epochs_checkpoint,
    )
    trainer.fit(module, train_dataloader=train_loader, val_dataloader=val_loader)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Per-epoch loss histograms: {LOSS_TRACKING_DIR}/epoch_*_loss_hist.png")
    print(f"Per-epoch slide losses:    {LOSS_TRACKING_DIR}/epoch_*_per_slide.csv")
    print(f"Epoch tracking table:      {LOSS_TRACKING_DIR}/loss_tracking_epochs.csv")
    print(f"Latest high-loss export:   {HIGH_LOSS_CSV}")
    print()
    print("Look for bimodal split in loss histograms while val_dice_mean plateaus.")
    print("Then pick epoch (data-driven, not a fixed number):")
    print("  python src/suggest_noise_epoch.py")
    print("  python src/suggest_noise_epoch.py --export")
    print("  python src/check_noise_overlap.py --epoch <N> --top-pct 10")


if __name__ == "__main__":
    main()
