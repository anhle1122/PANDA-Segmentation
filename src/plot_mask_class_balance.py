"""Aggregate mask pixel-class balance before vs after QC (read-only)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
MASKS = PROJECT / "data" / "masks"
OUT = PROJECT / "outputs"
FLAGS = OUT / "clean_dataset_flags.csv"
CLEAN = PROJECT / "data" / "radboud_clean.csv"

CLASS_NAMES = ["background", "stroma", "benign", "G3", "G4", "G5"]


def count_mask_classes(mask_path: Path, level: str = "pyramid") -> np.ndarray:
    """Return length-6 bincount for classes 0-5."""
    s = openslide.OpenSlide(str(mask_path))
    try:
        if level == "L0":
            w, h = s.level_dimensions[0]
            counts = np.zeros(6, dtype=np.int64)
            step = 4096
            for y in range(0, h, step):
                th = min(step, h - y)
                for x in range(0, w, step):
                    tw = min(step, w - x)
                    tile = np.array(s.read_region((x, y), 0, (tw, th)))[:, :, 0]
                    bc = np.bincount(tile.ravel(), minlength=6)
                    counts[: len(bc)] += bc[:6]
            return counts
        lvl = s.level_count - 1
        w, h = s.level_dimensions[lvl]
        arr = np.array(s.read_region((0, 0), lvl, (w, h)))[:, :, 0]
        bc = np.bincount(arr.ravel(), minlength=6)
        return bc[:6].astype(np.int64)
    finally:
        s.close()


def aggregate(ids: list[str], level: str) -> np.ndarray:
    total = np.zeros(6, dtype=np.int64)
    missing = 0
    for sid in tqdm(ids, desc=f"aggregate ({level})"):
        p = MASKS / f"{sid}_mask.tiff"
        if not p.exists():
            missing += 1
            continue
        total += count_mask_classes(p.resolve(), level=level)
    if missing:
        print(f"  skipped {missing} missing masks")
    return total


def plot_balance(pre: np.ndarray, post: np.ndarray, level: str, elapsed: float) -> None:
    pre_pct = 100.0 * pre / pre.sum()
    post_pct = 100.0 * post / post.sum()
    x = np.arange(6)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, pre_pct, w, label=f"Before QC (n={pre.sum():,.0f} px)", color="#4C72B0")
    ax.bar(x + w / 2, post_pct, w, label=f"After QC (n={post.sum():,.0f} px)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylabel("% of all mask pixels")
    ax.set_title(f"Mask pixel class balance — before vs after QC\n(read at {level})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i in x:
        ax.text(i - w / 2, pre_pct[i] + 0.2, f"{pre_pct[i]:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, post_pct[i] + 0.2, f"{post_pct[i]:.1f}%", ha="center", fontsize=8)
    note = f"Runtime: {elapsed/60:.1f} min at {level}"
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    png = OUT / "mask_class_balance_before_after.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved {png}")

    tbl = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "pixels_before": pre,
            "pct_before": np.round(pre_pct, 4),
            "pixels_after": post,
            "pct_after": np.round(post_pct, 4),
            "pct_point_change": np.round(post_pct - pre_pct, 4),
        }
    )
    csv = OUT / "mask_class_balance_before_after.csv"
    tbl.to_csv(csv, index=False)
    print(f"Saved {csv}")
    print(tbl.to_string(index=False))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["pyramid", "L0"], default="pyramid")
    p.add_argument("--benchmark", type=int, default=0, help="Benchmark N slides only")
    args = p.parse_args()

    flags = pd.read_csv(FLAGS)
    all_ids = flags["image_id"].tolist()
    clean_ids = set(pd.read_csv(CLEAN)["image_id"])
    post_ids = [i for i in all_ids if i in clean_ids]

    if args.benchmark:
        all_ids = all_ids[: args.benchmark]
        post_ids = [i for i in post_ids if i in set(all_ids)]

    print(f"Before QC slides: {len(all_ids)} | After QC: {len(post_ids)}")
    t0 = time.time()
    pre = aggregate(all_ids, args.level)
    post = aggregate(post_ids, args.level)
    elapsed = time.time() - t0
    plot_balance(pre, post, args.level, elapsed)

    meta = {
        "level": args.level,
        "slides_before": len(all_ids),
        "slides_after": len(post_ids),
        "elapsed_sec": elapsed,
        "per_slide_sec": elapsed / max(len(all_ids) + len(post_ids), 1),
    }
    (OUT / "mask_class_balance_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
