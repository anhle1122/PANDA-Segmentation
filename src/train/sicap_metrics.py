"""SICAPv2 scoring that treats mask 0 as non-cancer, not ignore/background.

Matches ``evaluate_sicapv2.dice_from_cm``: remap pred {1,2}→0, full 6×6 CM,
G3/G4/G5 Dice counts NC→cancer as FP. Do not use ignore_index=0 here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

SICAP_CLASSES = (0, 3, 4, 5)
SICAP_NAMES = {0: "NC", 3: "G3", 4: "G4", 5: "G5"}


def remap_panda_pred(pred: np.ndarray) -> np.ndarray:
    """Collapse PANDA {0,1,2} → SICAP NC=0; keep G3/G4/G5."""
    out = np.asarray(pred).copy()
    out[(out == 1) | (out == 2)] = 0
    return out


def accumulate_sicap_cm(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Add one batch into a 6×6 CM indexed [true, pred]. Mutates and returns cm."""
    pred = remap_panda_pred(np.asarray(pred).ravel())
    gt = np.clip(np.asarray(gt).ravel(), 0, 5).astype(np.int64)
    pred = np.clip(pred, 0, 5).astype(np.int64)
    idx = gt * 6 + pred
    cm += np.bincount(idx, minlength=36).reshape(6, 6)
    return cm


def dice_from_sicap_cm(cm: np.ndarray, classes: tuple[int, ...] = SICAP_CLASSES) -> dict[str, float]:
    """Per-class / cancer / binary Dice from a 6×6 confusion matrix."""
    out: dict[str, float] = {}
    for c in classes:
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        den = 2 * tp + fp + fn
        out[SICAP_NAMES[c]] = (2 * tp / den) if den > 0 else float("nan")
    vals = [out[SICAP_NAMES[c]] for c in classes if not np.isnan(out[SICAP_NAMES[c]])]
    out["mean4"] = float(np.mean(vals)) if vals else float("nan")
    cancer = [out[SICAP_NAMES[c]] for c in (3, 4, 5) if not np.isnan(out[SICAP_NAMES[c]])]
    out["cancer"] = float(np.mean(cancer)) if cancer else float("nan")
    tp = float(cm[np.ix_((3, 4, 5), (3, 4, 5))].sum())
    fp = float(cm[0, (3, 4, 5)].sum())
    fn = float(cm[(3, 4, 5), 0].sum())
    den = 2 * tp + fp + fn
    out["binary_cancer"] = (2 * tp / den) if den > 0 else float("nan")
    nc = float(cm[0, :].sum())
    nc_to_ca = float(cm[0, (3, 4, 5)].sum())
    out["nc_to_cancer"] = (nc_to_ca / nc) if nc > 0 else float("nan")
    return out


def sicap_cm_from_maps(pred: Tensor | np.ndarray, gt: Tensor | np.ndarray) -> np.ndarray:
    cm = np.zeros((6, 6), dtype=np.int64)
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    return accumulate_sicap_cm(cm, pred, gt)


def metrics_to_train_log(m: dict[str, float]) -> dict[str, float]:
    """Map SICAP CM metrics onto the trainer log schema (cancer_dice / mean_dice)."""
    cancer = float(m.get("cancer", 0.0) or 0.0)
    mean4 = float(m.get("mean4", 0.0) or 0.0)
    return {
        "dice_0": float(m.get("NC", 0.0) or 0.0),
        "dice_1": 0.0,
        "dice_2": 0.0,
        "dice_3": float(m.get("G3", 0.0) or 0.0),
        "dice_4": float(m.get("G4", 0.0) or 0.0),
        "dice_5": float(m.get("G5", 0.0) or 0.0),
        "mean_dice": mean4,
        "cancer_dice": cancer,
        "binary_cancer": float(m.get("binary_cancer", 0.0) or 0.0),
        "nc_to_cancer": float(m.get("nc_to_cancer", 0.0) or 0.0),
    }
