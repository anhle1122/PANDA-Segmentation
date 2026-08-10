"""Shared histology augmentations: patch-level and slide-consistent."""

from __future__ import annotations

from dataclasses import dataclass

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
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
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


class FixedHEDShift(ImageOnlyTransform):
    """Deterministic HED shift (same deltas for every patch in a slide bag)."""

    def __init__(
        self,
        dh: float,
        de: float,
        dd: float = 0.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.dh = float(dh)
        self.de = float(de)
        self.dd = float(dd)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        hed = rgb2hed(img.astype(np.float64) / 255.0)
        hed[..., 0] += self.dh
        hed[..., 1] += self.de
        if hed.shape[-1] > 2:
            hed[..., 2] += self.dd
        out = hed2rgb(np.clip(hed, 0, None))
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("dh", "de", "dd")


@dataclass(frozen=True)
class SlideAugParams:
    """One draw of slide-level aug applied identically to all patches in a bag."""

    dh: float
    de: float
    dd: float
    hflip: bool
    vflip: bool
    rotate90: int  # 0,1,2,3


def sample_slide_aug_params(
    rng: np.random.Generator | None = None,
    *,
    hematoxylin: float = 0.05,
    eosin: float = 0.05,
    dab: float = 0.02,
) -> SlideAugParams:
    rng = rng or np.random.default_rng()
    return SlideAugParams(
        dh=float(rng.uniform(-hematoxylin, hematoxylin)),
        de=float(rng.uniform(-eosin, eosin)),
        dd=float(rng.uniform(-dab, dab)),
        hflip=bool(rng.random() < 0.5),
        vflip=bool(rng.random() < 0.5),
        rotate90=int(rng.integers(0, 4)),
    )


def apply_slide_consistent(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    weight: np.ndarray | None,
    params: SlideAugParams,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply the same HED + flips + 90° rot to image (and optional mask/weight)."""
    hed = FixedHEDShift(params.dh, params.de, params.dd, p=1.0)
    rgb = hed.apply(rgb)
    if mask is not None:
        mask = mask.copy()
    if weight is not None:
        weight = weight.copy()
    if params.hflip:
        rgb = np.ascontiguousarray(np.fliplr(rgb))
        if mask is not None:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if weight is not None:
            weight = np.ascontiguousarray(np.fliplr(weight))
    if params.vflip:
        rgb = np.ascontiguousarray(np.flipud(rgb))
        if mask is not None:
            mask = np.ascontiguousarray(np.flipud(mask))
        if weight is not None:
            weight = np.ascontiguousarray(np.flipud(weight))
    if params.rotate90:
        k = int(params.rotate90) % 4
        rgb = np.ascontiguousarray(np.rot90(rgb, k))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask, k))
        if weight is not None:
            weight = np.ascontiguousarray(np.rot90(weight, k))
    return rgb, mask, weight


def build_train_augmentor() -> A.Compose:
    """Independent patch-level aug (Teacher A style)."""
    return A.Compose(
        [
            HEDShift(hematoxylin=0.05, eosin=0.05, dab=0.02, p=0.9),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        ],
        additional_targets={"weight": "mask"},
    )


def build_patch_extra_augmentor() -> A.Compose:
    """Light per-patch aug stacked after slide-consistent transforms."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
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


def apply_image_only(augmentor: A.Compose | None, rgb: np.ndarray) -> np.ndarray:
    if augmentor is None:
        return rgb
    return augmentor(image=rgb)["image"]
