"""WSISegQC pen.pt inference — slide thumbnail + patch-level (512×512) masks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import openslide
import segmentation_models_pytorch as smp
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_PEN_PT = PROJECT / "external" / "wsisegqc" / "models" / "models" / "pen.pt"
DEFAULT_MIN_PEN_PX = 10
DEFAULT_FLAG_PEN_PCT = 0.05  # % of tissue area on slide thumb


def pen_thumb_scale(slide: openslide.OpenSlide) -> tuple[float, float, int]:
    w, h = slide.dimensions
    magni = int(float(slide.properties.get("aperio.AppMag", "40")))
    ds = max(1, int(magni / 5))
    pen_w, pen_h = max(1, w // (ds * 8)), max(1, h // (ds * 8))
    return w / pen_w, h / pen_h, ds


@lru_cache(maxsize=1)
def _pen_model(weights: str, device: str):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def predict_pen_mask(rgb: np.ndarray, *, device: str, weights: Path = DEFAULT_PEN_PT) -> np.ndarray:
    """Run pen.pt on an RGB uint8 H×W×3 image; return binary pen mask (class 1)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected H×W×3 RGB, got {rgb.shape}")
    image = Image.fromarray(rgb.astype(np.uint8))
    model = _pen_model(str(weights), device)
    with torch.no_grad():
        w_, h_ = image.size
        wa, ha = w_ % 32, h_ % 32
        canvas = Image.new("RGB", (w_ + (32 - wa) + 128, h_ + (32 - ha) + 128))
        canvas.paste(image, (64, 64, w_ + 64, h_ + 64))
        arr = np.moveaxis(np.array(canvas), -1, 0)
        tensor = (torch.tensor(arr, dtype=torch.float32) / 255.0) - 0.5
        tensor = tensor.unsqueeze(0).to(device)
        pred = torch.argmax(model(tensor)[0], dim=0).cpu().numpy()
        pred = pred[64 : 64 + h_, 64 : 64 + w_]
    return (pred > 0).astype(np.uint8)


def patch_pen_metrics(
    pen_mask: np.ndarray,
    *,
    min_pen_px: int = 1,
) -> dict:
    n = pen_mask.size
    pen_px = int(pen_mask.sum())
    return {
        "pen_px": pen_px,
        "pen_pct_patch": 100.0 * pen_px / n if n else 0.0,
        "pen_flagged": pen_px >= min_pen_px,
    }


def infer_patch_pen(
    rgb: np.ndarray,
    *,
    device: str = "cpu",
    weights: Path = DEFAULT_PEN_PT,
    min_pen_px: int = DEFAULT_MIN_PEN_PX,
) -> dict:
    mask = predict_pen_mask(rgb, device=device, weights=weights)
    out = patch_pen_metrics(mask, min_pen_px=min_pen_px)
    out["pen_mask"] = mask
    return out


def infer_slide_pen_thumb(
    slide_path: Path,
    *,
    device: str = "cpu",
    weights: Path = DEFAULT_PEN_PT,
    min_pen_px: int = DEFAULT_MIN_PEN_PX,
    flag_pen_pct: float = DEFAULT_FLAG_PEN_PCT,
) -> dict:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        w, h = slide.dimensions
        sx, sy, _ = pen_thumb_scale(slide)
        pen_w, pen_h = int(round(w / sx)), int(round(h / sy))
        thumb = np.array(slide.get_thumbnail((pen_w, pen_h)))[:, :, :3]
        pen_mask = predict_pen_mask(thumb, device=device, weights=weights)
        tissue_npz = None
        npz_path = PROJECT / "outputs" / "pen_mark_detection_v2" / "npz" / f"{slide_path.stem}.npz"
        if npz_path.exists():
            tissue_npz = np.load(npz_path)["tissue"]
        tissue_mask = (
            (tissue_npz > 0).astype(np.uint8)
            if tissue_npz is not None
            else np.ones_like(pen_mask, dtype=np.uint8)
        )
        tissue_px = int(tissue_mask.sum())
        pen_px = int((pen_mask & tissue_mask).sum())
        pen_pct = 100.0 * pen_px / tissue_px if tissue_px else 0.0
        return {
            "pen_thumb_shape": pen_mask.shape,
            "scale_l0_x": sx,
            "scale_l0_y": sy,
            "tissue_area_px": tissue_px,
            "pen_area_px": pen_px,
            "pen_pct_of_tissue": pen_pct,
            "flagged": pen_px >= min_pen_px and pen_pct >= flag_pen_pct,
            "pen_mask": pen_mask,
            "tissue_mask": tissue_mask,
        }
    finally:
        slide.close()


def thumb_region_for_patch(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    patch_size: int,
    pen_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    sx, sy, _ = pen_thumb_scale(slide)
    x0, y0 = int(x / sx), int(y / sy)
    x1 = min(pen_mask.shape[1], int((x + patch_size) / sx) + 1)
    y1 = min(pen_mask.shape[0], int((y + patch_size) / sy) + 1)
    x0 = max(0, x0)
    y0 = max(0, y0)
    return pen_mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def infer_patch_from_slide_npz(
    slide_path: Path,
    x: int,
    y: int,
    patch_size: int = 512,
    *,
    min_pen_px: int = 1,
) -> dict:
    """Map L0 patch coords onto stored slide pen-thumb npz (not full-res pen.pt)."""
    npz_path = PROJECT / "outputs" / "pen_mark_detection_v2" / "npz" / f"{slide_path.stem}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    pen_mask = data["pen"]
    slide = openslide.OpenSlide(str(slide_path))
    try:
        region, bbox = thumb_region_for_patch(slide, x, y, patch_size, pen_mask)
        sx, sy, _ = pen_thumb_scale(slide)
        expected_thumb_px = max(1, int(round(patch_size / sx)) * int(round(patch_size / sy)))
        pen_px = int(region.sum())
        return {
            "source": "slide_npz_thumb",
            "thumb_bbox": bbox,
            "thumb_region_shape": region.shape,
            "pen_px": pen_px,
            "pen_pct_thumb_region": 100.0 * pen_px / region.size if region.size else 0.0,
            "pen_pct_patch_equiv": 100.0 * pen_px / expected_thumb_px if expected_thumb_px else 0.0,
            "pen_flagged": pen_px >= min_pen_px,
            "pen_mask_thumb": region,
        }
    finally:
        slide.close()
