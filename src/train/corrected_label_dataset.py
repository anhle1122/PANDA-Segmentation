"""Alternate label source: ISUP-referee corrected h5 (not Rules 1-3).

``apply_isup_referee.py`` writes per slide:

    <slide_id>_corrected.h5
        coords  (N, 2) int32
        target  (N, 512, 512) uint8   -- mask after nearest-allowed swaps
        ignore  (N, 512, 512) uint8   -- 1 = low-conf disagree, drop from loss

Slides with no file (ISUP-0, agree-only) fall back to the original mask.
Rules 1-3 (``PseudoLabelPatchDataset``) are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from train.baseline_dataset import BaselinePatchDataset
from train.losses import apply_pixel_ignore
from train.pseudo_label_dataset import PseudoLabelPatchDataset

CORRECTED_SUFFIX = "_corrected.h5"
LABEL_SOURCE_RULES = "rules"
LABEL_SOURCE_CORRECTED = "corrected"


class CorrectedLabelReader:
    """Random access into referee ``*_corrected.h5`` by (slide_id, x, y)."""

    def __init__(self, corrected_dir: str | Path, *, suffix: str = CORRECTED_SUFFIX) -> None:
        self.corrected_dir = Path(corrected_dir)
        self.suffix = suffix
        self._handles: dict[str, h5py.File] = {}
        self._index: dict[str, dict[tuple[int, int], int]] = {}

    def path_for(self, slide_id: str) -> Path:
        return self.corrected_dir / f"{slide_id}{self.suffix}"

    def has(self, slide_id: str) -> bool:
        return self.path_for(slide_id).exists()

    def read(self, slide_id: str, x: int, y: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (target uint8 HxW, ignore uint8 HxW) or None if missing."""
        if slide_id not in self._handles:
            path = self.path_for(slide_id)
            if not path.exists():
                return None
            handle = h5py.File(str(path), "r")
            self._handles[slide_id] = handle
            coords = handle["coords"][:]
            self._index[slide_id] = {
                (int(cx), int(cy)): i for i, (cx, cy) in enumerate(coords)
            }
        idx = self._index[slide_id].get((int(x), int(y)))
        if idx is None:
            return None
        h = self._handles[slide_id]
        return np.asarray(h["target"][idx]), np.asarray(h["ignore"][idx])

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._index.clear()


def overlay_corrected(
    mask: np.ndarray,
    weight: np.ndarray,
    target: np.ndarray,
    ignore: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use referee target; zero weight on ignore==1 pixels."""
    out_m = np.asarray(target).copy()
    out_w = np.asarray(weight, dtype=np.float32).copy()
    ign = np.asarray(ignore).astype(bool)
    out_m[ign] = 0
    out_w[ign] = 0.0
    _ = mask  # original mask unused when a correction exists
    return out_m, out_w


class RefereeCorrectedPatchDataset(Dataset):
    """Baseline patches with referee target + ignore folded into mask/weight.

    Returns ``(image, mask, weight)`` like ``BaselinePatchDataset`` so Opt3
    ``load_patch_batch`` can consume it later. Missing slide files keep the
    original mask (ISUP-0 / no-op).
    """

    def __init__(
        self,
        split_csv: str | Path,
        *,
        corrected_dir: str | Path,
        **baseline_kwargs,
    ) -> None:
        self.base = BaselinePatchDataset(split_csv, **baseline_kwargs)
        self.reader = CorrectedLabelReader(corrected_dir)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        image_t, mask_t, weight_t = self.base[idx]
        row = self.base.df.iloc[idx]
        slide_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])
        got = self.reader.read(slide_id, x, y)
        if got is None:
            return image_t, mask_t, weight_t
        target, ignore = got
        mask_np, weight_np = overlay_corrected(
            mask_t.numpy(), weight_t.numpy(), target, ignore
        )
        return (
            image_t,
            torch.from_numpy(mask_np.astype(np.int64)),
            torch.from_numpy(weight_np.astype(np.float32)),
        )

    def __del__(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()


def build_label_dataset(label_source: str, split_csv: str | Path, **kwargs):
    """Factory: ``rules`` keeps PseudoLabelPatchDataset; ``corrected`` uses referee h5."""
    source = str(label_source).strip().lower()
    if source == LABEL_SOURCE_RULES:
        return PseudoLabelPatchDataset(split_csv, **kwargs)
    if source == LABEL_SOURCE_CORRECTED:
        corrected_dir = kwargs.pop("corrected_dir", None)
        if not corrected_dir:
            raise ValueError("label_source=corrected requires corrected_dir=")
        baseline_kwargs = {
            k: kwargs[k]
            for k in kwargs
            if k
            not in {
                "manifest_csv",
                "pred_dir",
                "seg_target_dir",
                "allow_missing_cache",
            }
        }
        return RefereeCorrectedPatchDataset(
            split_csv, corrected_dir=corrected_dir, **baseline_kwargs
        )
    raise ValueError(f"Unknown label_source={label_source!r} (use 'rules' or 'corrected')")


def apply_ignore_to_loss_tensors(
    targets: Tensor,
    weight_map: Tensor,
    ignore: Tensor,
    *,
    ignore_index: int = 0,
) -> tuple[Tensor, Tensor]:
    """Thin wrapper so train code and tests share one path."""
    return apply_pixel_ignore(
        targets, weight_map, ignore, ignore_index=ignore_index
    )
