"""Model 2 dataset wrapper: adds per-patch Rule B/C pseudo-label flags on top
of BaselinePatchDataset, driven by the manifest from train.pseudo_label_rules.

Both Rule B's weight multiplier and Rule C's hard retarget need the model's
LIVE prediction (computed in the loss function, at training time) -- the
dataset only ever supplies cheap, static per-slide flags (is this slide Rule
B? Rule C_pure? what's its true class?), never precomputed pixel-level
targets. See train/losses.py combined_model2_loss for where the actual
per-pixel logic happens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from train.baseline_dataset import BaselinePatchDataset
from train.pseudo_label_rules import DEFAULT_MANIFEST_OUT, primary_gleason_class


class Model2PseudoLabelDataset(Dataset):
    """Wraps BaselinePatchDataset; returns (image, mask, weight, is_rule_b,
    is_rule_c_pure, rule_c_true_class) per patch."""

    def __init__(
        self,
        split_csv: str | Path,
        *,
        rule_manifest_csv: str | Path = DEFAULT_MANIFEST_OUT,
        **baseline_kwargs,
    ) -> None:
        self.base = BaselinePatchDataset(split_csv, **baseline_kwargs)
        manifest = pd.read_csv(rule_manifest_csv, dtype={"slide_id": str})
        self.rule_by_slide: dict[str, str] = dict(
            zip(manifest["slide_id"], manifest["rule"])
        )
        self.true_class_by_slide: dict[str, int] = {
            sid: primary_gleason_class(g)
            for sid, g, rule in zip(
                manifest["slide_id"], manifest["metadata_gleason"], manifest["rule"]
            )
            if rule == "C_pure"
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image_t, mask_t, weight_t = self.base[idx]
        slide_id = str(self.base.df.iloc[idx]["image_id"])
        rule = self.rule_by_slide.get(slide_id, "match")

        is_rule_b = torch.tensor(rule == "B", dtype=torch.bool)
        is_rule_c_pure = torch.tensor(rule == "C_pure", dtype=torch.bool)
        rule_c_true_class = torch.tensor(
            self.true_class_by_slide.get(slide_id, 0), dtype=torch.long,
        )

        return image_t, mask_t, weight_t, is_rule_b, is_rule_c_pure, rule_c_true_class

    def rule_counts(self) -> dict[str, int]:
        slide_ids = self.base.df["image_id"].astype(str)
        rules = slide_ids.map(lambda s: self.rule_by_slide.get(s, "match"))
        return rules.value_counts().to_dict()
