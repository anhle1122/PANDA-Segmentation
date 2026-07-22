"""Validation metrics — mean Dice over foreground Gleason classes."""

from __future__ import annotations

import torch
from torch import Tensor

from patch_utils import CLASS_NAMES

FOREGROUND_CLASS_NAMES = {k: v for k, v in CLASS_NAMES.items() if k > 0}


def dice_per_class(
    preds: Tensor,
    targets: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
    eps: float = 1e-6,
) -> Tensor:
    """Hard Dice per foreground class, shape (num_fg,)."""
    preds = preds.long()
    targets = targets.long()
    dice_scores = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = preds == cls
        target_c = targets == cls
        valid = targets != ignore_index
        pred_c = pred_c & valid
        target_c = target_c & valid
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    if not dice_scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(dice_scores)


def mean_foreground_dice(
    preds: Tensor,
    targets: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
) -> Tensor:
    """Scalar mean Dice over classes 1..C-1."""
    scores = dice_per_class(
        preds,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return scores.mean()


class PerClassDiceAccumulator:
    """Accumulate hard Dice intersection/union across a full validation epoch."""

    def __init__(self, *, num_classes: int = 6, ignore_index: int = 0) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        n = self.num_classes
        self.intersection = torch.zeros(n, dtype=torch.float64)
        self.union = torch.zeros(n, dtype=torch.float64)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        preds = preds.detach().long().cpu()
        targets = targets.detach().long().cpu()
        valid = targets != self.ignore_index
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_c = (preds == cls) & valid
            target_c = (targets == cls) & valid
            self.intersection[cls] += (pred_c & target_c).sum().double()
            self.union[cls] += pred_c.sum().double() + target_c.sum().double()

    def compute(self, *, eps: float = 1e-6) -> dict[str, float]:
        out: dict[str, float] = {}
        fg_scores = []
        for cls, name in FOREGROUND_CLASS_NAMES.items():
            if self.union[cls] > 0:
                dice = float((2.0 * self.intersection[cls] + eps) / (self.union[cls] + eps))
            else:
                dice = float("nan")
            out[name] = dice
            if not torch.isnan(torch.tensor(dice)):
                fg_scores.append(dice)
        out["mean"] = float(sum(fg_scores) / len(fg_scores)) if fg_scores else float("nan")
        return out
