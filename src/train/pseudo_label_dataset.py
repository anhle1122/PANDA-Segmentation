"""Dataset wrapper that attaches per-pixel pseudo-label corrections to patches.

Combines two inputs that were computed once, before training started:

  1. the slide-level rule manifest from train/pseudo_label_rules.py -- which
     rule applies, which source-model predicted classes to correct, and what
     to correct them to
  2. the cached per-pixel source-model predictions from
     scripts/cache_source_predictions.py

A pixel is flagged iff the SOURCE model predicted one of the slide's
``flag_pred_classes`` there. Neither input depends on the model currently
being trained, so the correction is stable for the whole round.

The target distribution is identical across every flagged pixel of a slide, so
instead of materializing a (C, H, W) float target per patch (~6 MB each) the
dataset returns a bool mask plus four scalars and lets
:func:`build_corrected_target` expand them on the GPU at loss time.

``seg_target_dir`` implements the round-over-round seg_target switch: leave it
None for Round 1 (target = the original PANDA mask) and point it at a cached
prediction directory for Round 2+ (target = the previous round's model's raw
argmax). The original mask file on disk is never modified either way.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from patch_utils import PROJECT
from train.baseline_dataset import BaselinePatchDataset
from train.pseudo_label_rules import MATCH, NO_ACTION, NO_CORRECTION_RULES

DEFAULT_MANIFEST = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"
DEFAULT_PRED_DIR = PROJECT / "outputs" / "pseudo_label" / "round1_source_pred"
CACHE_SUFFIX = "_srcpred.h5"

SEG_TARGET_MASK = "original_mask"
SEG_TARGET_MODEL = "model_prediction"

# Sentinel for "this sample has no correction" / "no cushion class".
NO_CLASS = -1


def parse_flag_classes(value) -> tuple[int, ...]:
    """'3|4' -> (3, 4). Blank/NaN -> ()."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(int(part) for part in text.split("|") if part != "")


class SlideRule:
    """One slide's correction parameters, resolved from the manifest."""

    __slots__ = ("flag_classes", "main_class", "main_weight", "cushion_class", "cushion_weight")

    def __init__(
        self,
        flag_classes: tuple[int, ...],
        main_class: int,
        main_weight: float,
        cushion_class: int,
        cushion_weight: float,
    ) -> None:
        self.flag_classes = flag_classes
        self.main_class = main_class
        self.main_weight = main_weight
        self.cushion_class = cushion_class
        self.cushion_weight = cushion_weight


class PredictionCacheReader:
    """Random access into the per-slide HDF5 prediction caches, by (x, y).

    Handles are opened lazily so each DataLoader worker gets its own, matching
    how BaselinePatchDataset manages its H5/openslide handles.
    """

    def __init__(self, cache_dir: str | Path, *, suffix: str = CACHE_SUFFIX) -> None:
        self.cache_dir = Path(cache_dir)
        self.suffix = suffix
        self._handles: dict[str, h5py.File] = {}
        self._index: dict[str, dict[tuple[int, int], int]] = {}

    def path_for(self, slide_id: str) -> Path:
        return self.cache_dir / f"{slide_id}{self.suffix}"

    def has(self, slide_id: str) -> bool:
        return self.path_for(slide_id).exists()

    def read(self, slide_id: str, x: int, y: int) -> np.ndarray | None:
        """Predicted class map for one patch, or None if not cached."""
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
        idx = self._index[slide_id].get((x, y))
        if idx is None:
            return None
        return self._handles[slide_id]["preds"][idx]

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._index.clear()


def build_corrected_target(
    flag_mask: Tensor,
    target_params: Tensor,
    *,
    num_classes: int = 6,
) -> Tensor:
    """Expand per-sample target scalars into the (B, C, H, W) tensor combined_loss wants.

    ``target_params`` is (B, 4): main_class, main_weight, cushion_class,
    cushion_weight. Rows whose main_class is NO_CLASS contribute nothing --
    those samples have no flagged pixels anyway.

    Values off the flagged pixels are left at zero; combined_loss only ever
    reads the target where flag_mask is True.
    """
    B, H, W = flag_mask.shape
    device = flag_mask.device
    target = torch.zeros(B, num_classes, H, W, device=device, dtype=torch.float32)

    main_class = target_params[:, 0].long()
    main_weight = target_params[:, 1].to(torch.float32)
    cushion_class = target_params[:, 2].long()
    cushion_weight = target_params[:, 3].to(torch.float32)

    flag_f = flag_mask.to(torch.float32)
    for b in range(B):
        mc = int(main_class[b])
        if mc == NO_CLASS:
            continue
        target[b, mc] = flag_f[b] * float(main_weight[b])
        cc = int(cushion_class[b])
        if cc != NO_CLASS:
            target[b, cc] = flag_f[b] * float(cushion_weight[b])
    return target


class PseudoLabelPatchDataset(Dataset):
    """BaselinePatchDataset plus (flag_mask, target_params) per patch.

    Returns ``(image, seg_target, weight, flag_mask, target_params)`` where
    ``seg_target`` is the original mask or a cached model prediction depending
    on ``seg_target_dir``, ``flag_mask`` is (H, W) bool, and ``target_params``
    is [main_class, main_weight, cushion_class, cushion_weight].
    """

    def __init__(
        self,
        split_csv: str | Path,
        *,
        manifest_csv: str | Path = DEFAULT_MANIFEST,
        pred_dir: str | Path = DEFAULT_PRED_DIR,
        seg_target_dir: str | Path | None = None,
        allow_missing_cache: bool = False,
        **baseline_kwargs,
    ) -> None:
        self.base = BaselinePatchDataset(split_csv, **baseline_kwargs)
        self.allow_missing_cache = allow_missing_cache
        self.source_preds = PredictionCacheReader(pred_dir)
        self.seg_targets = (
            PredictionCacheReader(seg_target_dir) if seg_target_dir is not None else None
        )
        self.seg_target_mode = SEG_TARGET_MASK if seg_target_dir is None else SEG_TARGET_MODEL

        manifest = pd.read_csv(manifest_csv, dtype={"slide_id": str})
        self.rule_by_slide: dict[str, str] = dict(
            zip(manifest["slide_id"], manifest["rule_applied"])
        )
        self.slide_rules: dict[str, SlideRule] = {}
        corrected = manifest[~manifest["rule_applied"].isin(NO_CORRECTION_RULES)]
        for row in corrected.itertuples():
            flag_classes = parse_flag_classes(row.flag_pred_classes)
            if not flag_classes:
                continue
            self.slide_rules[str(row.slide_id)] = SlideRule(
                flag_classes=flag_classes,
                main_class=int(row.target_main_class),
                main_weight=float(row.target_main_weight),
                cushion_class=int(row.target_cushion_class),
                cushion_weight=float(row.target_cushion_weight),
            )

    def __len__(self) -> int:
        return len(self.base)

    def _no_correction(self, mask_t: Tensor) -> tuple[Tensor, Tensor]:
        flag_mask = torch.zeros_like(mask_t, dtype=torch.bool)
        params = torch.tensor([NO_CLASS, 0.0, NO_CLASS, 0.0], dtype=torch.float32)
        return flag_mask, params

    def __getitem__(self, idx: int):
        image_t, mask_t, weight_t = self.base[idx]
        row = self.base.df.iloc[idx]
        slide_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])

        if self.seg_targets is not None:
            cached = self.seg_targets.read(slide_id, x, y)
            if cached is None:
                if not self.allow_missing_cache:
                    raise FileNotFoundError(
                        f"seg_target_mode=model_prediction but no cached prediction for "
                        f"{slide_id} @ ({x},{y}) in {self.seg_targets.cache_dir}"
                    )
            else:
                mask_t = torch.from_numpy(cached.astype(np.int64))

        rule = self.slide_rules.get(slide_id)
        if rule is None:
            flag_mask, params = self._no_correction(mask_t)
            return image_t, mask_t, weight_t, flag_mask, params

        src_pred = self.source_preds.read(slide_id, x, y)
        if src_pred is None:
            if not self.allow_missing_cache:
                raise FileNotFoundError(
                    f"Missing source-prediction cache for corrected slide {slide_id} @ ({x},{y}) "
                    f"in {self.source_preds.cache_dir}. Run scripts/cache_source_predictions.py "
                    "first, or pass allow_missing_cache=True to train without this correction."
                )
            flag_mask, params = self._no_correction(mask_t)
            return image_t, mask_t, weight_t, flag_mask, params

        flagged = np.isin(src_pred, rule.flag_classes)
        flag_mask = torch.from_numpy(flagged)
        params = torch.tensor(
            [rule.main_class, rule.main_weight, rule.cushion_class, rule.cushion_weight],
            dtype=torch.float32,
        )
        return image_t, mask_t, weight_t, flag_mask, params

    def rule_counts(self) -> dict[str, int]:
        """Rule breakdown over the patches in this split (not slides)."""
        slide_ids = self.base.df["image_id"].astype(str)
        rules = slide_ids.map(lambda s: self.rule_by_slide.get(s, MATCH))
        return rules.value_counts().to_dict()

    def __del__(self) -> None:
        for reader in (getattr(self, "source_preds", None), getattr(self, "seg_targets", None)):
            if reader is not None:
                reader.close()
