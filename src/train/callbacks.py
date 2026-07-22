"""Training callbacks: per-sample loss histograms and high-loss slide export."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from train.losses import dice_ce_loss_per_sample


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient. > ~0.555 suggests bimodal separation."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 4:
        return 0.0
    mean = values.mean()
    std = values.std(ddof=1)
    if std <= 0:
        return 0.0
    m3 = np.mean((values - mean) ** 3)
    m4 = np.mean((values - mean) ** 4)
    skew = m3 / (std ** 3)
    kurt = m4 / (std ** 4)
    numer = (skew ** 2) + 1.0
    denom = kurt + ((3.0 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))) - 3.0
    if denom == 0:
        return 0.0
    return float(numer / denom)


class PerSampleLossCallback(Callback):
    """Collect per-patch train losses, plot histograms, export per-slide high-loss CSV."""

    def __init__(
        self,
        train_loader,
        output_dir: Path,
        every_n_epochs: int = 1,
        high_loss_csv: Path | None = None,
        min_epoch_for_export: int = 5,
        tail_std_multiplier: float = 1.5,
    ) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.high_loss_csv = high_loss_csv
        self.min_epoch_for_export = min_epoch_for_export
        self.tail_std_multiplier = tail_std_multiplier
        self.epoch_metrics = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        if self.train_loader is None:
            return

        device = pl_module.device
        pl_module.eval()
        patch_losses = []
        image_ids = []

        with torch.no_grad():
            for batch in self.train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = pl_module(images)
                per_sample = dice_ce_loss_per_sample(
                    logits,
                    masks,
                    num_classes=pl_module.hparams.num_classes,
                    ignore_index=pl_module.hparams.ignore_index,
                )
                patch_losses.extend(per_sample.detach().cpu().numpy().tolist())
                image_ids.extend(list(batch["image_id"]))

        losses = np.asarray(patch_losses, dtype=np.float64)
        df = pd.DataFrame({"image_id": image_ids, "patch_loss": patch_losses})

        slide_df = (
            df.groupby("image_id", as_index=False)["patch_loss"]
            .agg(mean_loss="mean", max_loss="max", n_patches="count")
            .sort_values("mean_loss", ascending=False)
        )
        slide_df["loss_rank"] = np.arange(1, len(slide_df) + 1)
        slide_df["epoch"] = epoch

        mean = losses.mean()
        std = losses.std(ddof=1) if losses.size > 1 else 0.0
        tail_threshold = mean + self.tail_std_multiplier * std
        tail_count = int((losses > tail_threshold).sum())
        bimod = bimodality_coefficient(losses)

        val_dice = float(trainer.callback_metrics.get("val_dice_mean", torch.tensor(float("nan"))))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch_tag = f"epoch_{epoch:03d}"
        hist_path = self.output_dir / f"{epoch_tag}_loss_hist.png"
        slide_path = self.output_dir / f"{epoch_tag}_per_slide.csv"
        patch_path = self.output_dir / f"{epoch_tag}_per_patch.csv"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(losses, bins=50, color="steelblue", edgecolor="white")
        ax.axvline(tail_threshold, color="crimson", linestyle="--", label=f"tail threshold ({tail_threshold:.3f})")
        ax.set_title(f"Per-patch train loss — epoch {epoch}")
        ax.set_xlabel("Dice+CE loss (per patch)")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)

        slide_df.to_csv(slide_path, index=False)
        df.to_csv(patch_path, index=False)

        metrics = {
            "epoch": epoch,
            "n_patches": int(losses.size),
            "n_slides": int(slide_df.shape[0]),
            "loss_mean": round(float(mean), 6),
            "loss_std": round(float(std), 6),
            "tail_threshold": round(float(tail_threshold), 6),
            "tail_patch_count": tail_count,
            "tail_patch_pct": round(100.0 * tail_count / max(losses.size, 1), 2),
            "bimodality_coefficient": round(bimod, 4),
            "val_dice_mean": None if np.isnan(val_dice) else round(val_dice, 4),
            "hist_path": str(hist_path),
            "per_slide_path": str(slide_path),
        }
        self.epoch_metrics.append(metrics)

        metrics_csv = self.output_dir / "loss_tracking_epochs.csv"
        pd.DataFrame(self.epoch_metrics).to_csv(metrics_csv, index=False)

        if self.high_loss_csv is not None and epoch >= self.min_epoch_for_export:
            slide_df.to_csv(self.high_loss_csv, index=False)

        pl_module.log("train_loss_bimodality", bimod, on_epoch=True, prog_bar=False)
        pl_module.log("train_loss_tail_pct", metrics["tail_patch_pct"], on_epoch=True, prog_bar=False)

        summary_path = self.output_dir / "loss_tracking_summary.json"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        pl_module.eval()
