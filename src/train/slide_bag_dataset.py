"""Slide-bag dataset for Option 3 (MIL-style) training.

Regroups existing patch rows in ``panda_train.csv`` by ``image_id``. Each
``__getitem__`` returns **one slide**: a variable-length bag of patches plus
the clinician ISUP label. No WSI re-extraction.

The training loop is expected to micro-batch the bag (e.g. 4–8 patches) and
accumulate gradients, then compute slide-level ISUP losses once the full bag
has been painted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from patch_utils import PROJECT
from train.baseline_dataset import BaselinePatchDataset

DEFAULT_METADATA = PROJECT / "data" / "train.csv"
DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"


class SlideBagPatchDataset(Dataset):
    """One item = one slide bag.

    Returns a dict:
      image_id: str
      images:   FloatTensor (N, 3, H, W)
      masks:    LongTensor  (N, H, W)
      weights:  FloatTensor (N, H, W)
      isup:     LongTensor  scalar clinician ISUP (0–5)
      coords:   LongTensor  (N, 2)  optional debug
    """

    def __init__(
        self,
        split_csv: str | Path = DEFAULT_SPLIT,
        *,
        metadata_csv: str | Path = DEFAULT_METADATA,
        max_patches_per_slide: int | None = None,
        seed: int = 42,
        **baseline_kwargs,
    ) -> None:
        self.base = BaselinePatchDataset(split_csv, **baseline_kwargs)
        self.max_patches_per_slide = max_patches_per_slide
        self.rng = np.random.default_rng(seed)

        meta = pd.read_csv(metadata_csv, dtype={"image_id": str})
        if "isup_grade" not in meta.columns:
            raise ValueError(f"{metadata_csv} missing isup_grade column")
        self.isup_by_slide = dict(
            zip(meta["image_id"].astype(str), meta["isup_grade"].astype(int))
        )

        df = self.base.df.copy()
        df["image_id"] = df["image_id"].astype(str)
        groups: dict[str, list[int]] = {}
        for i, sid in enumerate(df["image_id"].tolist()):
            groups.setdefault(sid, []).append(i)
        # Stable slide order
        self.slide_ids = sorted(groups.keys())
        self.indices_by_slide = {s: groups[s] for s in self.slide_ids}

        missing = [s for s in self.slide_ids if s not in self.isup_by_slide]
        if missing:
            raise ValueError(
                f"{len(missing)} slides in split lack isup_grade in {metadata_csv} "
                f"(e.g. {missing[:3]})"
            )

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        slide_id = self.slide_ids[idx]
        patch_idxs = list(self.indices_by_slide[slide_id])
        if (
            self.max_patches_per_slide is not None
            and len(patch_idxs) > self.max_patches_per_slide
        ):
            chosen = self.rng.choice(
                len(patch_idxs), size=self.max_patches_per_slide, replace=False
            )
            patch_idxs = [patch_idxs[i] for i in sorted(chosen.tolist())]

        images, masks, weights, coords = [], [], [], []
        for pi in patch_idxs:
            image_t, mask_t, weight_t = self.base[pi]
            row = self.base.df.iloc[pi]
            images.append(image_t)
            masks.append(mask_t)
            weights.append(weight_t)
            coords.append((int(row["x"]), int(row["y"])))

        return {
            "image_id": slide_id,
            "images": torch.stack(images, dim=0),
            "masks": torch.stack(masks, dim=0),
            "weights": torch.stack(weights, dim=0),
            "isup": torch.tensor(int(self.isup_by_slide[slide_id]), dtype=torch.long),
            "coords": torch.tensor(coords, dtype=torch.long),
        }


def slide_bag_collate(batch: list[dict]) -> dict:
    """Collate a list of slide bags (usually batch_size=1)."""
    if len(batch) != 1:
        # Multi-slide batches are possible but uncommon; keep simple for now.
        raise ValueError(
            f"SlideBag collate expects batch_size=1 (got {len(batch)} slides). "
            "Use micro-batches inside the training loop over each slide's patches."
        )
    return batch[0]


def summarize_bags(dataset: SlideBagPatchDataset) -> dict:
    sizes = [len(dataset.indices_by_slide[s]) for s in dataset.slide_ids]
    return {
        "n_slides": len(sizes),
        "n_patches": int(sum(sizes)),
        "min_patches": int(min(sizes)),
        "median_patches": float(np.median(sizes)),
        "max_patches": int(max(sizes)),
        "mean_patches": float(np.mean(sizes)),
    }
