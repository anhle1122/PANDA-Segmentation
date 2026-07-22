"""Median-frequency class weights from training-patch mask pixel counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from patch_utils import CLASS_NAMES, MASKS_DIR, NUM_CLASSES, PATCH_SIZE, PROJECT, read_label_patch

SPLITS_DIR = PROJECT / "outputs" / "splits"
FOREGROUND_CLASS_NAMES = {k: v for k, v in CLASS_NAMES.items() if k > 0}


def median_frequency_weights(pixel_counts: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """weight_c = median(freq) / freq_c over all classes."""
    counts = pixel_counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.ones(len(counts), dtype=np.float32)
    freq = counts / total
    median_freq = float(np.median(freq))
    weights = median_freq / (freq + eps)
    return weights.astype(np.float32)


def pixel_freq_cache_path(mode: str) -> Path:
    return SPLITS_DIR / f"panda_train_pixel_freq_{mode}.pt"


def compute_pixel_counts(
    patch_df: pd.DataFrame,
    *,
    masks_dir: Path = MASKS_DIR,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for row in tqdm(patch_df.itertuples(), total=len(patch_df), desc="pixel freq"):
        image_id = str(row.image_id)
        x, y = int(row.x), int(row.y)
        mask_path = masks_dir / f"{image_id}_mask.tiff"
        mask = read_label_patch(mask_path, x, y, PATCH_SIZE)
        counts += np.bincount(mask.ravel(), minlength=num_classes)[:num_classes]
    return counts


def build_class_weight_bundle(
    patch_df: pd.DataFrame,
    *,
    mode: str,
    masks_dir: Path = MASKS_DIR,
) -> dict:
    pixel_counts = compute_pixel_counts(patch_df, masks_dir=masks_dir)
    total_pixels = int(pixel_counts.sum())
    pixel_freq = pixel_counts.astype(np.float64) / max(total_pixels, 1)
    class_weights = median_frequency_weights(pixel_counts)
    return {
        "mode": mode,
        "pixel_counts": torch.tensor(pixel_counts, dtype=torch.int64),
        "pixel_freq": torch.tensor(pixel_freq, dtype=torch.float64),
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "n_patches": int(len(patch_df)),
        "total_pixels": total_pixels,
        "class_names": dict(CLASS_NAMES),
    }


def save_class_weight_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_class_weight_bundle(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_or_compute_class_weights(
    train_df: pd.DataFrame,
    *,
    mode: str,
    recompute: bool = False,
    masks_dir: Path = MASKS_DIR,
) -> dict:
    cache_path = pixel_freq_cache_path(mode)
    if cache_path.exists() and not recompute:
        bundle = load_class_weight_bundle(cache_path)
        if bundle.get("n_patches") == len(train_df) and bundle.get("mode") == mode:
            return bundle

    bundle = build_class_weight_bundle(train_df, mode=mode, masks_dir=masks_dir)
    save_class_weight_bundle(bundle, cache_path)
    return bundle


def format_weight_table(bundle: dict) -> str:
    lines = ["Class weights (median frequency):"]
    weights = bundle["class_weights"].numpy()
    freqs = bundle["pixel_freq"].numpy()
    counts = bundle["pixel_counts"].numpy()
    for idx, name in CLASS_NAMES.items():
        lines.append(
            f"  {name:10s}: weight={weights[idx]:.4f}  freq={freqs[idx]:.6f}  pixels={int(counts[idx])}"
        )
    return "\n".join(lines)
