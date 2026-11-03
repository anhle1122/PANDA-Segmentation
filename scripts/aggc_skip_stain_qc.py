#!/usr/bin/env python3
"""CPU QC: empty-tile skip vs annotations, plus Macenko thumbnail stain vs tissue.

No GPU. Writes stain_skip_qc.csv and stain_vs_tissue.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path("/common/omarmlab/members/anh/panda_project")
sys.path.insert(0, str(ROOT / "src"))

from evaluate_aggc2022 import (  # noqa: E402
    DEFAULT_ROOT,
    MASK_PRIORITY,
    MIN_SLIDE_TISSUE_PIXELS,
    TILE_SKIP_MEAN,
    TILE_SKIP_STD,
    estimate_slide_stain_matrix,
    find_ann_dir,
    find_image_path,
    list_slides,
)
from stain_norm_sicap import PANDA_REFERENCE  # noqa: E402
from stain_normalize import get_normalizer, load_reference_patch  # noqa: E402


def build_gt_10x(ann_dir: Path, slide_id: str, w0: int, h0: int) -> np.ndarray:
    w2, h2 = -(-w0 // 10), -(-h0 // 10)
    gt = np.zeros((h2, w2), dtype=np.uint8)
    slide_dir = find_ann_dir(ann_dir, slide_id)
    if slide_dir is None:
        return gt
    for fname, class_idx in MASK_PRIORITY:
        mpath = slide_dir / fname
        if not mpath.is_file():
            continue
        im = Image.open(mpath)
        small = im.resize((w2, h2), resample=Image.NEAREST)
        arr = np.array(small)
        if arr.ndim == 3:
            arr = arr[..., 0]
        gt[arr > 0] = class_idx
        im.close()
    return gt


def skip_vs_annotation(
    img: Image.Image, gt10: np.ndarray, *, tile: int = 512
) -> dict:
    w0, h0 = img.size
    n_total = n_skip = skip_ann = 0
    for y in range(0, h0, tile):
        for x in range(0, w0, tile):
            n_total += 1
            box = (x, y, min(x + tile, w0), min(y + tile, h0))
            t = np.array(img.crop(box).convert("RGB"))
            if not (t.mean() > TILE_SKIP_MEAN and t.std() < TILE_SKIP_STD):
                continue
            n_skip += 1
            y2, x2 = y // 10, x // 10
            h2 = max(1, (box[3] - box[1] + 9) // 10)
            w2 = max(1, (box[2] - box[0] + 9) // 10)
            patch = gt10[y2 : y2 + h2, x2 : x2 + w2]
            skip_ann += int((patch > 0).sum())
    return {
        "tiles_total": n_total,
        "tiles_skipped": n_skip,
        "skip_ann_px_10x": skip_ann,
    }


def plot_stain(rows: list[dict], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Subset1": "#1f77b4", "Subset2": "#ff7f0e", "Subset3": "#2ca02c"}
    for subset, color in colors.items():
        xs, ys, fail_x, fail_y = [], [], [], []
        for r in rows:
            if r["subset"] != subset:
                continue
            tf = float(r["tissue_frac"])
            tp = float(r["thumb_tissue_px"])
            if int(r["stain_ok"]):
                xs.append(tp)
                ys.append(tf)
            else:
                fail_x.append(tp)
                fail_y.append(tf)
        if xs:
            ax.scatter(xs, ys, c=color, label=f"{subset} fit OK", s=28, alpha=0.85)
        if fail_x:
            ax.scatter(
                fail_x,
                fail_y,
                c=color,
                marker="x",
                s=40,
                label=f"{subset} below {MIN_SLIDE_TISSUE_PIXELS} px",
            )
    ax.axvline(MIN_SLIDE_TISSUE_PIXELS, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Thumbnail tissue pixels (LAB mask)")
    ax.set_ylabel("Tissue fraction of thumbnail")
    ax.set_title("Macenko slide-stain thumbnail vs tissue area")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggc-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--groups",
        default="Subset1/Test,Subset2/Test,Subset3/Test",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/evaluation/aggc2022/qc_skip_stain",
    )
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--skip-check", action="store_true", default=True)
    p.add_argument("--no-skip-check", action="store_true")
    args = p.parse_args()
    do_skip = not args.no_skip_check

    ref = load_reference_patch(PANDA_REFERENCE)
    normalizer, _backend = get_normalizer("macenko")
    normalizer.fit(ref)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for group in args.groups.split(","):
        subset, split = group.strip().split("/")
        names = list_slides(args.aggc_root, subset, split)
        img_dir = args.aggc_root / subset / split / "images"
        ann_dir = args.aggc_root / subset / split / "annotations"
        print(f"=== {subset}/{split} n={len(names)} ===", flush=True)
        for i, sid in enumerate(names, 1):
            img_path = find_image_path(img_dir, sid)
            if img_path is None:
                print(f"  WARN missing image {sid}", flush=True)
                continue
            img = Image.open(img_path)
            w0, h0 = img.size
            stain = estimate_slide_stain_matrix(normalizer, img)
            rec = {
                "subset": subset,
                "split": split,
                "slide_id": sid,
                "w": w0,
                "h": h0,
                "stain_ok": int(stain["ok"]),
                "thumb_tissue_px": stain["thumb_tissue_px"],
                "thumb_px": stain["thumb_px"],
                "tissue_frac": stain["tissue_frac"],
                "stain_h_norm": stain["stain_h_norm"],
                "stain_e_norm": stain["stain_e_norm"],
                "tiles_total": "",
                "tiles_skipped": "",
                "skip_ann_px_10x": "",
                "ann_px_10x": "",
            }
            if do_skip:
                gt10 = build_gt_10x(ann_dir, sid, w0, h0)
                skip = skip_vs_annotation(img, gt10, tile=args.tile)
                rec.update({k: skip[k] for k in skip})
                rec["ann_px_10x"] = int((gt10 > 0).sum())
            rows.append(rec)
            img.close()
            print(
                f"  [{i}/{len(names)}] {sid} stain_ok={rec['stain_ok']} "
                f"tissue_px={rec['thumb_tissue_px']} frac={rec['tissue_frac']:.4f} "
                f"skip_ann={rec.get('skip_ann_px_10x', '')}",
                flush=True,
            )

    csv_path = args.out_dir / "stain_skip_qc.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    png = args.out_dir / "stain_vs_tissue.png"
    plot_stain(rows, png)
    n_bad_skip = sum(
        1 for r in rows if r.get("skip_ann_px_10x") not in ("", None) and int(r["skip_ann_px_10x"]) > 0
    )
    n_stain_fail = sum(1 for r in rows if not int(r["stain_ok"]))
    summary = {
        "n": len(rows),
        "slides_with_annotated_pixels_in_skipped_tiles": n_bad_skip,
        "stain_fit_failed": n_stain_fail,
        "min_slide_tissue_pixels": MIN_SLIDE_TISSUE_PIXELS,
        "csv": str(csv_path),
        "plot": str(png),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
