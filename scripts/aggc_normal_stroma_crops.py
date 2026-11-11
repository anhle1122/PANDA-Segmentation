#!/usr/bin/env python3
"""Two crops of pred-stroma inside AGGC Normal: between glands vs on glands.

Omar: if stroma sits between benign glands → annotation-granularity mismatch;
if on the glands themselves → benign detection problem under domain shift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/common/omarmlab/members/anh/panda_project")
sys.path.insert(0, str(ROOT / "src"))

from evaluate import (  # noqa: E402
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    detect_arch,
    load_model_weights,
)
from evaluate_aggc2022 import (  # noqa: E402
    DEFAULT_ROOT,
    build_gt_index_map,
    downsize_10x,
    find_ann_dir,
    find_image_path,
    infer_index_map,
)


def epithelium_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristic: purple/dark gland epithelium vs pink stroma."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    # hematoxylin-ish: darker, blue/purple not eosin-pink
    epi = (bright < 175) & (b >= r * 0.85) & (b >= g * 0.9)
    return epi


def pink_stroma_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    return (r > 150) & (r > b + 15) & (bright > 140) & (bright < 230)


def largest_cc(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return mask of largest connected component and its area (4-connected)."""
    from scipy import ndimage  # type: ignore

    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    i = int(counts.argmax())
    return lab == i, int(counts[i])


def crop_around(mask: np.ndarray, rgb: np.ndarray, pad: int = 80) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    return rgb[y0:y1, x0:x1], (y0, y1, x0, x1)


def paint_panel(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, focus: np.ndarray, title: str, out: Path) -> None:
    panel = rgb.copy()
    normal = gt == 2
    inside = normal & (pred == 1)
    # faint yellow all Normal
    panel[normal] = (0.65 * panel[normal] + np.array([220, 200, 40]) * 0.35).astype(np.uint8)
    # magenta all stroma-in-Normal
    panel[inside] = (0.35 * panel[inside] + np.array([200, 30, 160]) * 0.65).astype(np.uint8)
    # cyan outline of the focus subtype
    from scipy import ndimage  # type: ignore

    edge = focus & ~ndimage.binary_erosion(focus, iterations=1)
    panel[edge] = np.array([0, 255, 255], dtype=np.uint8)
    im = Image.fromarray(panel)
    # upscale small crops for readability
    w, h = im.size
    scale = max(1, int(800 / max(w, h)))
    if scale > 1:
        im = im.resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.size[0], 44), fill=(0, 0, 0))
    draw.text((8, 12), title, fill=(255, 255, 255))
    im.save(out)
    print(f"wrote {out} crop={w}x{h} scale={scale}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--slide", default="Subset1_Test_12")
    p.add_argument("--aggc-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-tiles", type=int, default=4)
    args = p.parse_args()

    import torch

    device = torch.device("cuda")
    resolved = detect_arch(args.checkpoint, "auto")
    _ckpt, keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()

    sid = args.slide
    subset = "Subset1"
    split = "Test"
    img_dir = args.aggc_root / subset / split / "images"
    ann_dir = args.aggc_root / subset / split / "annotations"
    img_path = find_image_path(img_dir, sid)
    if img_path is None:
        raise SystemExit(f"missing image {sid}")
    img = Image.open(img_path)
    w0, h0 = img.size
    gt = build_gt_index_map(ann_dir, sid, (h0, w0))
    pred, _ = infer_index_map(
        model, img, device=device, amp=True, scale_factor=1.0,
        tile=512, batch_tiles=args.batch_tiles, gt_native=gt,
    )
    gt2, pred2 = downsize_10x(gt), downsize_10x(pred)
    rgb = np.array(img.resize((gt2.shape[1], gt2.shape[0]), Image.BILINEAR).convert("RGB"))
    img.close()

    normal = gt2 == 2
    stroma_in = normal & (pred2 == 1)
    epi = epithelium_mask(rgb)
    pink = pink_stroma_mask(rgb)

    # on glands: stroma prediction that lands on epithelium-looking pixels inside Normal
    on_gland = stroma_in & epi
    # between glands: stroma prediction on pink fibrous look inside Normal, not epithelium
    between = stroma_in & pink & ~epi

    n_norm = int(normal.sum())
    n_in = int(stroma_in.sum())
    n_on = int(on_gland.sum())
    n_be = int(between.sum())
    print(
        f"{sid}: Normal={n_norm} stroma_in={n_in} ({n_in/n_norm:.3f}) "
        f"on_gland={n_on} ({n_on/max(n_in,1):.3f} of stroma_in) "
        f"between={n_be} ({n_be/max(n_in,1):.3f} of stroma_in)",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # save overview thumb
    overview = rgb.copy()
    overview[normal] = (0.65 * overview[normal] + np.array([220, 200, 40]) * 0.35).astype(np.uint8)
    overview[stroma_in] = (0.3 * overview[stroma_in] + np.array([200, 30, 160]) * 0.7).astype(np.uint8)
    Image.fromarray(overview).resize((1600, int(1600 * overview.shape[0] / overview.shape[1])), Image.BILINEAR).save(
        args.out_dir / f"{sid}_overview_stroma_in_normal.png"
    )

    for name, mask in (("between_glands", between), ("on_glands", on_gland)):
        if mask.sum() < 50:
            print(f"WARN: too few pixels for {name}: {mask.sum()}", flush=True)
            continue
        cc, area = largest_cc(mask)
        crop_rgb, box = crop_around(cc, rgb, pad=100)
        # also crop gt/pred for paint
        y0, y1, x0, x1 = box
        paint_panel(
            crop_rgb,
            gt2[y0:y1, x0:x1],
            pred2[y0:y1, x0:x1],
            cc[y0:y1, x0:x1],
            title=f"{sid} {name} (cyan=focus) yellow=GT Normal magenta=pred stroma area={area}",
            out=args.out_dir / f"{sid}_crop_{name}.png",
        )

    meta = args.out_dir / f"{sid}_crop_stats.txt"
    meta.write_text(
        f"slide={sid}\nNormal_px={n_norm}\nstroma_in_Normal={n_in}\n"
        f"frac_stroma_in_Normal={n_in/max(n_norm,1):.4f}\n"
        f"on_gland_px={n_on}\nbetween_glands_px={n_be}\n"
        f"frac_of_stroma_in_that_is_on_gland={n_on/max(n_in,1):.4f}\n"
        f"frac_of_stroma_in_that_is_between={n_be/max(n_in,1):.4f}\n"
        "heuristic: epithelium=dark purple; between=pink eosin inside Normal∩pred_stroma\n"
    )
    print(f"wrote {meta}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
