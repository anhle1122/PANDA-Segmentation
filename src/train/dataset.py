"""PyTorch Dataset for PANDA 512x512 patch segmentation."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augmentation_preview import HEDShift
from patch_utils import (
    MASKS_DIR,
    NUM_CLASSES,
    PATCH_SIZE,
    SLIDES_DIR,
    read_label_patch,
    read_rgb_patch,
)


def build_train_augmentor() -> A.Compose:
    return A.Compose(
        [
            HEDShift(hematoxylin=0.05, eosin=0.05, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )


def build_eval_augmentor() -> A.Compose:
    return A.Compose([], additional_targets={"mask": "mask"})


class PandaPatchDataset(Dataset):
    """Loads RGB (WSI or H5) + mask patches indexed by patch_index rows."""

    def __init__(
        self,
        patch_df: pd.DataFrame,
        *,
        augment: bool = False,
        slides_dir: Path = SLIDES_DIR,
        masks_dir: Path = MASKS_DIR,
    ) -> None:
        self.patch_df = patch_df.reset_index(drop=True)
        self.augment = augment
        self.slides_dir = slides_dir
        self.masks_dir = masks_dir
        self.transform = build_train_augmentor() if augment else build_eval_augmentor()
        self._h5_handles: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.patch_df)

    def _read_image(self, row: pd.Series) -> np.ndarray:
        h5_path = str(row.get("h5_path", "") or "")
        h5_index = int(row.get("h5_index", -1))
        if h5_path and h5_index >= 0 and Path(h5_path).exists():
            if h5_path not in self._h5_handles:
                self._h5_handles[h5_path] = h5py.File(h5_path, "r")
            return self._h5_handles[h5_path]["imgs"][h5_index]
        image_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])
        slide_path = self.slides_dir / f"{image_id}.tiff"
        return read_rgb_patch(slide_path, x, y, PATCH_SIZE)

    def __getitem__(self, idx: int) -> dict:
        row = self.patch_df.iloc[idx]
        image_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])

        image = self._read_image(row)
        mask_path = self.masks_dir / f"{image_id}_mask.tiff"
        mask = read_label_patch(mask_path, x, y, PATCH_SIZE)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return {
            "image": image_t,
            "mask": mask_t,
            "image_id": image_id,
            "x": x,
            "y": y,
        }

    def __del__(self) -> None:
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass


def split_patch_index(
    patch_df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by image_id so patches from the same slide stay in one fold."""
    slide_ids = sorted(patch_df["image_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(slide_ids)
    n_val = max(1, int(round(len(slide_ids) * val_fraction)))
    val_ids = set(slide_ids[:n_val])
    train_df = patch_df[~patch_df["image_id"].astype(str).isin(val_ids)].copy()
    val_df = patch_df[patch_df["image_id"].astype(str).isin(val_ids)].copy()
    return train_df, val_df


def load_patch_index(
    patch_index_csv: Path,
    *,
    clean_slides_csv: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(patch_index_csv)
    if clean_slides_csv and clean_slides_csv.exists():
        clean_ids = set(pd.read_csv(clean_slides_csv)["image_id"].astype(str))
        df = df[df["image_id"].astype(str).isin(clean_ids)].copy()
    return df


def subsample_patch_index(patch_df: pd.DataFrame, max_patches: int, *, seed: int = 42) -> pd.DataFrame:
    if len(patch_df) <= max_patches:
        return patch_df.copy()
    return patch_df.sample(n=max_patches, random_state=seed).reset_index(drop=True)
