"""PyTorch Lightning module for PANDA Gleason patch segmentation."""

from __future__ import annotations

import torch
from pytorch_lightning import LightningModule

from train.losses import dice_ce_loss, dice_ce_loss_per_sample
from train.metrics import FOREGROUND_CLASS_NAMES, PerClassDiceAccumulator
from train.model import UNetSmall


class GleasonSegmentationModule(LightningModule):
    def __init__(
        self,
        *,
        num_classes: int = 6,
        ignore_index: int = 0,
        lr: float = 1e-3,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        base_channels: int = 32,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = UNetSmall(
            in_channels=3,
            num_classes=num_classes,
            base=base_channels,
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self._val_dice_acc = PerClassDiceAccumulator(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss_kwargs(self) -> dict:
        return {
            "num_classes": self.hparams.num_classes,
            "ignore_index": self.hparams.ignore_index,
            "ce_weight": self.hparams.ce_weight,
            "dice_weight": self.hparams.dice_weight,
            "class_weight": self.class_weights,
        }

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch["image"])
        targets = batch["mask"]
        loss = dice_ce_loss(logits, targets, **self._loss_kwargs())
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if stage == "val":
            preds = logits.argmax(dim=1)
            self._val_dice_acc.update(preds, targets)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_dice_acc.reset()

    def on_validation_epoch_end(self) -> None:
        scores = self._val_dice_acc.compute()
        for name in FOREGROUND_CLASS_NAMES.values():
            value = scores.get(name, float("nan"))
            self.log(
                f"val_dice_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name in {"G5", "mean"}),
            )
        self.log(
            "val_dice_mean",
            scores["mean"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice_mean",
            },
        }

    @torch.no_grad()
    def predict_per_sample_loss(self, batch: dict) -> tuple[torch.Tensor, list[str]]:
        logits = self(batch["image"])
        per_sample = dice_ce_loss_per_sample(
            logits,
            batch["mask"],
            **self._loss_kwargs(),
        )
        return per_sample, list(batch["image_id"])
