"""Suggest which epoch to use for high-loss / noise overlap analysis.

There is no universal "epoch 5" — pick the epoch where BOTH signals align:
  1. val_dice_mean has largely stopped improving (plateau)
  2. Per-patch train loss histogram is splitting into two humps (bimodal)

Reads outputs/loss_tracking/loss_tracking_epochs.csv written during training.

Usage:
  python src/suggest_noise_epoch.py
  python src/suggest_noise_epoch.py --export   # copy that epoch -> high_loss_slides.csv
  python src/check_noise_overlap.py --epoch <N>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
LOSS_TRACKING_DIR = PROJECT / "outputs" / "loss_tracking"
EPOCHS_CSV = LOSS_TRACKING_DIR / "loss_tracking_epochs.csv"
HIGH_LOSS_CSV = PROJECT / "outputs" / "high_loss_slides.csv"
SUGGESTION_JSON = LOSS_TRACKING_DIR / "suggested_noise_epoch.json"

# Sarle's bimodality coefficient: > ~0.555 suggests two humps
BIMODAL_THRESHOLD = 0.555
# Val Dice gain below this over a 2-epoch window counts as plateau
DICE_PLATEAU_DELTA = 0.01
# Need at least this many epochs before trusting plateau detection
MIN_EPOCHS_BEFORE_PLATEAU = 3


def detect_val_dice_plateau(df: pd.DataFrame) -> int | None:
    """First epoch (1-indexed) where val Dice improvement stalls."""
    if "val_dice_mean" not in df.columns:
        return None
    dice = df["val_dice_mean"].dropna()
    if len(dice) < MIN_EPOCHS_BEFORE_PLATEAU + 1:
        return None

    epochs = df.loc[dice.index, "epoch"].astype(int).tolist()
    values = dice.tolist()
    for i in range(MIN_EPOCHS_BEFORE_PLATEAU - 1, len(values) - 1):
        gain_next = values[i + 1] - values[i]
        if i + 2 < len(values):
            gain_after = values[i + 2] - values[i + 1]
        else:
            gain_after = gain_next
        if gain_next < DICE_PLATEAU_DELTA and gain_after < DICE_PLATEAU_DELTA:
            return int(epochs[i + 1])
    return int(epochs[-1])


def detect_bimodal_epoch(df: pd.DataFrame) -> int | None:
    """First epoch where per-patch loss distribution looks bimodal."""
    if "bimodality_coefficient" not in df.columns:
        return None
    for _, row in df.sort_values("epoch").iterrows():
        bimod = row["bimodality_coefficient"]
        if pd.notna(bimod) and float(bimod) >= BIMODAL_THRESHOLD:
            return int(row["epoch"])
    return None


def suggest_epoch(df: pd.DataFrame) -> dict:
    plateau_epoch = detect_val_dice_plateau(df)
    bimodal_epoch = detect_bimodal_epoch(df)

    candidates = [e for e in (plateau_epoch, bimodal_epoch) if e is not None]
    if plateau_epoch and bimodal_epoch:
        recommended = max(plateau_epoch, bimodal_epoch)
        rationale = (
            f"Use the later of val-Dice plateau (epoch {plateau_epoch}) and "
            f"bimodal loss split (epoch {bimodal_epoch}). "
            "Before both signals align, high-loss rankings are not reliable."
        )
    elif bimodal_epoch:
        recommended = bimodal_epoch
        rationale = (
            f"Bimodal loss split at epoch {bimodal_epoch}. "
            "Val Dice plateau not detected yet — inspect loss histograms manually."
        )
    elif plateau_epoch:
        recommended = plateau_epoch
        rationale = (
            f"Val Dice plateau at epoch {plateau_epoch}. "
            "Bimodality not reached 0.555 yet — high-loss tail may still be noisy."
        )
    elif len(df):
        recommended = int(df["epoch"].max())
        rationale = (
            f"Insufficient signal — defaulting to last epoch ({recommended}). "
            "Review loss histograms manually."
        )
    else:
        recommended = None
        rationale = "No epoch metrics found."

    return {
        "recommended_epoch": recommended,
        "val_dice_plateau_epoch": plateau_epoch,
        "bimodal_split_epoch": bimodal_epoch,
        "bimodal_threshold": BIMODAL_THRESHOLD,
        "dice_plateau_delta": DICE_PLATEAU_DELTA,
        "rationale": rationale,
    }


def export_epoch_slides(epoch: int) -> Path:
    src = LOSS_TRACKING_DIR / f"epoch_{epoch:03d}_per_slide.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Train with --loss-hist-every 1 first.")
    shutil.copy2(src, HIGH_LOSS_CSV)
    return HIGH_LOSS_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest epoch for high-loss noise analysis")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Copy recommended epoch per-slide losses to outputs/high_loss_slides.csv",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Override recommended epoch (for --export)",
    )
    args = parser.parse_args()

    if not EPOCHS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {EPOCHS_CSV}. Run training first with per-epoch loss tracking."
        )

    df = pd.read_csv(EPOCHS_CSV)
    result = suggest_epoch(df)
    result["epochs_csv"] = str(EPOCHS_CSV)
    SUGGESTION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("SUGGESTED EPOCH FOR HIGH-LOSS / NOISE ANALYSIS")
    print("=" * 60)
    print(f"Val Dice plateau epoch:    {result['val_dice_plateau_epoch']}")
    print(f"Bimodal split epoch:       {result['bimodal_split_epoch']}  (threshold {BIMODAL_THRESHOLD})")
    print(f"Recommended epoch:         {result['recommended_epoch']}")
    print()
    print(result["rationale"])
    print()
    print("How to verify manually:")
    print(f"  1. Open {LOSS_TRACKING_DIR}/epoch_*_loss_hist.png — look for two humps")
    print("  2. Plot val_dice_mean from loss_tracking_epochs.csv — look for plateau")
    print("  3. Pick the epoch where both agree, then:")
    rec = result["recommended_epoch"]
    if rec:
        print(f"     python src/suggest_noise_epoch.py --export --epoch {rec}")
        print(f"     python src/check_noise_overlap.py --epoch {rec} --top-pct 10")
    print("=" * 60)

    export_epoch = args.epoch or result["recommended_epoch"]
    if args.export:
        if export_epoch is None:
            raise ValueError("No epoch to export.")
        out = export_epoch_slides(int(export_epoch))
        print(f"\nExported epoch {export_epoch} -> {out}")


if __name__ == "__main__":
    main()
