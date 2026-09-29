"""CPU gates for SICAP merged-NC loss (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from train.losses import sicap_merged_dice_loss, sicap_merged_nc_ce  # noqa: E402
from train.sicap_metrics import dice_from_sicap_cm, sicap_cm_from_maps  # noqa: E402


def _peaked_logits(cls: int, *, peak: float = 12.0) -> torch.Tensor:
    z = torch.zeros(1, 6, 1, 1)
    z[0, cls, 0, 0] = peak
    return z


def _nc_loss(pred_cls: int) -> float:
    logits = _peaked_logits(pred_cls)
    targets = torch.zeros(1, 1, 1, dtype=torch.long)
    weights = torch.ones(1, 1, 1)
    class_w = torch.ones(6)
    ce = sicap_merged_nc_ce(logits, targets, weights, class_w, adjacent_soft_alpha=0.1)
    dice = sicap_merged_dice_loss(logits, targets)
    return float((0.5 * ce + 0.5 * dice).item())


def main() -> None:
    loss_g4 = _nc_loss(4)
    loss_st = _nc_loss(1)
    loss_bg = _nc_loss(0)
    loss_ben = _nc_loss(2)
    print(
        f"NC-pixel losses: G4={loss_g4:.6f} stroma={loss_st:.6f} "
        f"bg={loss_bg:.6f} benign={loss_ben:.6f}"
    )
    if not (loss_g4 > loss_st + 0.5):
        raise SystemExit(
            f"GATE FAIL: G4-on-NC ({loss_g4:.4f}) should exceed stroma-on-NC "
            f"({loss_st:.4f}) by >0.5"
        )
    nc_preds = (loss_bg, loss_st, loss_ben)
    spread = max(nc_preds) - min(nc_preds)
    print(f"NC-class spread (bg/stroma/benign)={spread:.6f}")
    if spread > 0.05:
        raise SystemExit(
            f"GATE FAIL: bg/stroma/benign on NC should be ~equal, spread={spread:.4f}"
        )

    # Honest CM: G4 on NC is an FP for G4 Dice.
    pred = torch.tensor([[[[4]]]])
    gt = torch.tensor([[[[0]]]])
    cm = sicap_cm_from_maps(pred, gt)
    m = dice_from_sicap_cm(cm)
    if m["nc_to_cancer"] < 0.99:
        raise SystemExit(f"GATE FAIL: NC→cancer should be 1, got {m['nc_to_cancer']}")
    print("GATE_OK unit_test_sicap_nc_merge")


if __name__ == "__main__":
    main()
