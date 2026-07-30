"""Baseline training dataset — raw (A) and Vahadane-normalized (B) modes."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import Dataset

from patch_utils import MASKS_DIR, PATCH_SIZE, PROJECT, SLIDES_DIR
from stain_normalize import stain_artifact_mask
from train.data_index import RAW_H5_DIR, RAW_H5_STEM, TRAIN_MODES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

H5_IMAGE_KEY = "imgs"
H5_COORDS_KEY = "coords"


def _preprocess_image(rgb: np.ndarray) -> torch.Tensor:
    patch = rgb.astype(np.float32) / 255.0
    patch = (patch - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(patch.transpose(2, 0, 1))


class BaselinePatchDataset(Dataset):
    """Returns (image_tensor, mask_tensor, weight_tensor)."""

    def __init__(
        self,
        split_csv: str | Path,
        *,
        mode: str = "raw",
        h5_dir: str | Path | None = None,
        h5_stem: str | None = None,
        slide_dir: str | Path = SLIDES_DIR,
        mask_dir: str | Path = MASKS_DIR,
        allow_missing_h5: bool = False,
        mask_suffix: str = "_mask.tiff",
        prefer_h5_masks: bool = True,
    ) -> None:
        if mode not in TRAIN_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.df = pd.read_csv(split_csv)
        self.slide_dir = Path(slide_dir)
        self.mask_dir = Path(mask_dir)
        self.allow_missing_h5 = allow_missing_h5
        self.mask_suffix = mask_suffix
        self.prefer_h5_masks = prefer_h5_masks
        mode_cfg = TRAIN_MODES[mode]
        self.h5_dir = Path(h5_dir) if h5_dir is not None else mode_cfg["h5_dir"]
        self.h5_stem = h5_stem or mode_cfg["h5_stem"]
        self.raw_h5_dir = mode_cfg["raw_h5_dir"]
        self.raw_h5_stem = mode_cfg["raw_h5_stem"]
        self.zero_artifact_loss = bool(mode_cfg["zero_artifact_loss"])
        self._h5_handles: dict[str, h5py.File] = {}
        self._h5_index: dict[str, dict[tuple[int, int], int]] = {}
        self._mask_cache: dict[str, openslide.OpenSlide] = {}
        self._png_mask_cache: dict[str, np.ndarray] = {}
        self._slide_cache: dict[str, openslide.OpenSlide] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _h5_path(self, image_id: str, *, h5_dir: Path | None = None, h5_stem: str | None = None) -> Path:
        return (h5_dir or self.h5_dir) / f"{image_id}_{h5_stem or self.h5_stem}.h5"

    def _load_h5_index(self, image_id: str, *, h5_dir: Path, h5_stem: str) -> None:
        path = str(self._h5_path(image_id, h5_dir=h5_dir, h5_stem=h5_stem))
        if path in self._h5_index:
            return
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        coords = self._h5_handles[path][H5_COORDS_KEY][:]
        self._h5_index[path] = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}

    def _read_h5_patch(
        self,
        image_id: str,
        x: int,
        y: int,
        *,
        h5_dir: Path,
        h5_stem: str,
    ) -> np.ndarray:
        h5_path = str(self._h5_path(image_id, h5_dir=h5_dir, h5_stem=h5_stem))
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"Missing H5: {h5_path}")
        self._load_h5_index(image_id, h5_dir=h5_dir, h5_stem=h5_stem)
        idx = self._h5_index[h5_path][(x, y)]
        return self._h5_handles[h5_path][H5_IMAGE_KEY][idx]

    def _read_slide_patch(self, image_id: str, x: int, y: int) -> np.ndarray:
        if image_id not in self._slide_cache:
            self._slide_cache[image_id] = openslide.OpenSlide(str(self.slide_dir / f"{image_id}.tiff"))
        slide = self._slide_cache[image_id]
        return np.array(slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))[:, :, :3]

    def _read_image(self, image_id: str, x: int, y: int) -> np.ndarray:
        h5_path = self._h5_path(image_id)
        if h5_path.exists():
            try:
                return self._read_h5_patch(image_id, x, y, h5_dir=self.h5_dir, h5_stem=self.h5_stem)
            except KeyError:
                if not self.allow_missing_h5:
                    raise
                return self._read_slide_patch(image_id, x, y)
        if self.allow_missing_h5:
            return self._read_slide_patch(image_id, x, y)
        raise FileNotFoundError(f"Missing H5 for mode={self.mode}: {h5_path}")

    def _artifact_weight_map(self, image_id: str, x: int, y: int) -> np.ndarray:
        weight = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        if not self.zero_artifact_loss:
            return weight
        raw_path = self._h5_path(image_id, h5_dir=self.raw_h5_dir, h5_stem=self.raw_h5_stem)
        if raw_path.exists():
            raw = self._read_h5_patch(
                image_id, x, y, h5_dir=self.raw_h5_dir, h5_stem=self.raw_h5_stem,
            )
        elif self.allow_missing_h5:
            slide = openslide.OpenSlide(str(self.slide_dir / f"{image_id}.tiff"))
            try:
                raw = np.array(slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))[:, :, :3]
            finally:
                slide.close()
        else:
            return weight
        artifact = stain_artifact_mask(raw, mask_artifacts=True, mask_glass=True)
        weight[artifact] = 0.0
        return weight

    def _mask_file(self, image_id: str) -> Path:
        return self.mask_dir / f"{image_id}{self.mask_suffix}"

    def _read_png_mask_patch(self, image_id: str, x: int, y: int) -> np.ndarray:
        import cv2

        if image_id not in self._png_mask_cache:
            path = self._mask_file(image_id)
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Missing mask: {path}")
            self._png_mask_cache[image_id] = mask
        full = self._png_mask_cache[image_id]
        patch = full[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Mask crop OOB for {image_id} @ ({x},{y}): got {patch.shape}, "
                f"mask={full.shape}"
            )
        return np.clip(patch, 0, 5).astype(np.int64)

    def _read_mask(self, image_id: str, x: int, y: int) -> np.ndarray:
        if self.mask_suffix.lower().endswith(".png"):
            return self._read_png_mask_patch(image_id, x, y)
        if image_id not in self._mask_cache:
            mask_path = str(self._mask_file(image_id))
            self._mask_cache[image_id] = openslide.OpenSlide(mask_path)
        mask_slide = self._mask_cache[image_id]
        mask = np.array(mask_slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return np.clip(mask, 0, 5).astype(np.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])
        rgb = self._read_image(image_id, x, y)
        mask = self._read_mask(image_id, x, y)
        image_t = _preprocess_image(rgb)
        mask_t = torch.from_numpy(mask)
        weight_t = torch.from_numpy(self._artifact_weight_map(image_id, x, y))
        return image_t, mask_t, weight_t

    def run_sanity_check(self, n: int = 5) -> None:
        import random

        print(
            f"Sanity check — mode={self.mode}, h5_dir={self.h5_dir}, "
            f"zero_artifact_loss={self.zero_artifact_loss}, n={n}, patches={len(self)}"
        )
        for i in random.sample(range(len(self)), min(n, len(self))):
            image, mask, weight = self[i]
            if image.shape != (3, PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"Bad image shape: {image.shape}")
            if image.dtype != torch.float32:
                raise ValueError(f"Bad image dtype: {image.dtype}")
            if not (-4.0 < float(image.min()) < 4.0 and -4.0 < float(image.max()) < 4.0):
                raise ValueError(
                    f"Image values out of expected ImageNet range: "
                    f"[{float(image.min()):.3f}, {float(image.max()):.3f}]"
                )
            uniq = set(mask.unique().tolist())
            if not uniq.issubset({0, 1, 2, 3, 4, 5}):
                raise ValueError(f"Mask has invalid labels: {uniq}")
            if weight.shape != (PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"Bad weight shape: {weight.shape}")
            if self.zero_artifact_loss and torch.all(weight == 1.0):
                print(f"  [{i}] note: no artifact pixels in sample (weight all ones)")
            print(
                f"  [{i}] image={tuple(image.shape)} mask_labels={sorted(uniq)} "
                f"weight_zero_frac={float((weight == 0).mean()):.3f} "
                f"img_range=[{image.min():.2f},{image.max():.2f}]"
            )
        print("Sanity check PASSED")

    def __del__(self) -> None:
        for h in getattr(self, "_h5_handles", {}).values():
            try:
                h.close()
            except Exception:
                pass
        for slide in getattr(self, "_mask_cache", {}).values():
            try:
                slide.close()
            except Exception:
                pass
        if hasattr(self, "_png_mask_cache"):
            self._png_mask_cache.clear()
        for slide in self._slide_cache.values():
            try:
                slide.close()
            except Exception:
                pass
        self._png_mask_cache.clear()
