"""Shared histology augmentations for Gleason patch segmentation."""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import hed2rgb, rgb2hed


class HEDShift(ImageOnlyTransform):
    """Random perturbation in HED color space (histology-specific)."""

    def __init__(
        self,
        hematoxylin: float = 0.05,
        eosin: float = 0.05,
        dab: float = 0.02,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.hematoxylin = hematoxylin
        self.eosin = eosin
        self.dab = dab

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        hed = rgb2hed(img.astype(np.float64) / 255.0)
        hed[..., 0] += np.random.uniform(-self.hematoxylin, self.hematoxylin)
        hed[..., 1] += np.random.uniform(-self.eosin, self.eosin)
        if hed.shape[-1] > 2:
            hed[..., 2] += np.random.uniform(-self.dab, self.dab)
        out = hed2rgb(np.clip(hed, 0, None))
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("hematoxylin", "eosin", "dab")


def build_train_augmentor() -> A.Compose:
    """HED shift + flips + rotation; mask/weight stay label-aligned."""
    return A.Compose(
        [
            HEDShift(hematoxylin=0.05, eosin=0.05, dab=0.02, p=0.9),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        ],
        additional_targets={"weight": "mask"},
    )


def apply_augment(
    augmentor: A.Compose | None,
    rgb: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if augmentor is None:
        return rgb, mask, weight
    out = augmentor(image=rgb, mask=mask.astype(np.int32), weight=weight.astype(np.float32))
    return (
        out["image"],
        out["mask"].astype(np.int64),
        out["weight"].astype(np.float32),
    )
