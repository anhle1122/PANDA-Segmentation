#!/usr/bin/env python3
"""Overlay: model stroma inside AGGC Normal polygons (5 worst Normal Dice slides)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def colorize(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Yellow = GT Normal, magenta = pred stroma ∩ GT Normal, cyan = pred stroma elsewhere."""
    out = rgb.copy()
    normal = gt == 2
    stroma_pred = pred == 1
    inside = normal & stroma_pred
    # blend
    out[normal] = (0.55 * out[normal] + np.array([220, 200, 40]) * 0.45).astype(np.uint8)
    out[inside] = (0.25 * out[inside] + np.array([200, 30, 160]) * 0.75).astype(np.uint8)
    return out


def thumb(arr: np.ndarray, max_side: int = 1600) -> Image.Image:
    im = Image.fromarray(arr)
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--aggc-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--slides", required=True, help="slide_ids with +")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in args.slides.replace(",", "+").split("+") if s.strip()]
    for sid in ids:
        subset = "Subset1" if sid.startswith("Subset1") else (
            "Subset2" if sid.startswith("Subset2") else "Subset3"
        )
        split = "Test" if "_Test_" in sid or sid.startswith(f"{subset}_Test") else "Train"
        img_dir = args.aggc_root / subset / split / "images"
        ann_dir = args.aggc_root / subset / split / "annotations"
        img_path = find_image_path(img_dir, sid)
        if img_path is None:
            print(f"missing {sid}", flush=True)
            continue
        img = Image.open(img_path)
        w0, h0 = img.size
        gt = build_gt_index_map(ann_dir, sid, (h0, w0))
        pred, _ = infer_index_map(
            model, img, device=device, amp=True, scale_factor=1.0,
            tile=512, batch_tiles=args.batch_tiles, gt_native=gt,
        )
        gt2, pred2 = downsize_10x(gt), downsize_10x(pred)
        rgb = np.array(img.resize((gt2.shape[1], gt2.shape[0]), Image.BILINEAR).convert("RGB"))
        panel = colorize(rgb, gt2, pred2)
        n_norm = int((gt2 == 2).sum())
        n_in = int(((gt2 == 2) & (pred2 == 1)).sum())
        frac = n_in / n_norm if n_norm else float("nan")
        im = thumb(panel)
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, im.size[0], 36), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            f"{sid}  stroma-inside-Normal={frac:.3f}  yellow=GT Normal  magenta=our stroma in Normal",
            fill=(255, 255, 255),
        )
        out = args.out_dir / f"{sid}_stroma_in_normal.png"
        im.save(out)
        img.close()
        print(f"wrote {out} frac={frac:.4f} n_norm={n_norm} n_in={n_in}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
