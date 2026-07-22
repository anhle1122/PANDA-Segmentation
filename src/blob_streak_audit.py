"""Audit thick black-ink streak: why each tile was kept vs dropped.

Focuses on the vertical pen tracing along x≈5632–7680 on the blob slide.
Recomputes live HSV stats and compares to patch_filtering_v2 CSV decisions.

  sbatch scripts/slurm_blob_streak_audit.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageFilter

from patch_utils import PATCH_SIZE, PROJECT, SLIDES_DIR, read_rgb_patch, read_label_patch, MASKS_DIR

OTS_DIR = PROJECT / "outputs" / "trident_all" / "20x_512px_0px_overlap" / "patches"
REMOVED_CSV = PROJECT / "outputs" / "patch_filtering_v2" / "removed_patches.csv"
RESCUED_CSV = PROJECT / "outputs" / "patch_filtering_v2" / "rescued_patches.csv"
DEFAULT_OUT = PROJECT / "outputs" / "wholeslide_pipeline_compare"
DEFAULT_SLIDE = "00928370e2dfeb8a507667ef1d4efcbb"

TISSUE_THRESHOLD = 40.0
PEN_THRESHOLD = 2.0
BLUE_PEN_THRESHOLD = 5.0
WHITE_THRESHOLD = 190

# Thick vertical streak region (L0 coords) — user screenshot area
STREAK_X0, STREAK_Y0 = 5120, 2048
STREAK_X1, STREAK_Y1 = 8192, 12288


def pen_stats(patch_rgb: np.ndarray) -> dict:
    pil = Image.fromarray(patch_rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    arr = np.array(pil)
    white = np.all(arr > WHITE_THRESHOLD, axis=2)
    blank = (np.max(arr, axis=2) - np.min(arr, axis=2) < 20) & (np.mean(arr, axis=2) > 200)
    tissue_pct = 100.0 * (1.0 - (white | blank).sum() / (PATCH_SIZE * PATCH_SIZE))

    hsv = np.array(pil.convert("HSV"))
    blue = (
        (hsv[:, :, 0] >= 170) & (hsv[:, :, 0] <= 180)
        & (hsv[:, :, 1] >= 125) & (hsv[:, :, 2] >= 30)
    )
    black = (hsv[:, :, 0] <= 180) & ((hsv[:, :, 2] <= 150) | (hsv[:, :, 2] <= 30))
    green = (
        (hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 70)
        & (hsv[:, :, 1] >= 125) & (hsv[:, :, 2] >= 30)
    )
    area = PATCH_SIZE * PATCH_SIZE
    return {
        "tissue_pct": tissue_pct,
        "black_pct": 100.0 * black.sum() / area,
        "blue_pct": 100.0 * blue.sum() / area,
        "green_pct": 100.0 * green.sum() / area,
    }


def expected_decision(s: dict) -> str:
    if s["tissue_pct"] < TISSUE_THRESHOLD:
        return "drop_low_tissue"
    if s["blue_pct"] > BLUE_PEN_THRESHOLD:
        return "drop_pen_blue"
    if s["black_pct"] > PEN_THRESHOLD:
        return "drop_pen_black"
    if s["green_pct"] > PEN_THRESHOLD:
        return "drop_pen_green"
    return "keep"


def load_coords(slide_id: str) -> list[tuple[int, int]]:
    with h5py.File(OTS_DIR / f"{slide_id}_patches.h5", "r") as f:
        return [(int(x), int(y)) for x, y in f["coords"][:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", default=DEFAULT_SLIDE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sid = args.slide_id
    args.out_dir.mkdir(parents=True, exist_ok=True)

    removed = pd.read_csv(REMOVED_CSV)
    removed = removed[removed.slide_id == sid].set_index(["x", "y"])
    rescued = set(
        zip(
            pd.read_csv(RESCUED_CSV).query("slide_id == @sid").x.astype(int),
            pd.read_csv(RESCUED_CSV).query("slide_id == @sid").y.astype(int),
        )
    )

    streak_tiles = [
        (x, y) for x, y in load_coords(sid)
        if STREAK_X0 <= x < STREAK_X1 and STREAK_Y0 <= y < STREAK_Y1
    ]
    streak_tiles.sort(key=lambda t: (t[0], t[1]))

    rows = []
    for x, y in streak_tiles:
        rgb = read_rgb_patch(SLIDES_DIR / f"{sid}.tiff", x, y)
        mask = read_label_patch(MASKS_DIR / f"{sid}_mask.tiff", x, y)
        s = pen_stats(rgb)
        exp = expected_decision(s)
        cancer = int(((mask >= 3) & (mask <= 5)).sum())
        if (x, y) in rescued.index if hasattr(rescued, 'index') else (x, y) in rescued:
            final = "RESCUED"
        elif (x, y) in removed.index:
            final = f"DROP {removed.loc[(x, y), 'reason']}"
        else:
            final = "KEEP"
        rows.append({"x": x, "y": y, "final": final, "expected": exp, "cancer_px": cancer, **s})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # High-res crop of the streak
    slide = openslide.OpenSlide(str(SLIDES_DIR / f"{sid}.tiff"))
    cw, ch = STREAK_X1 - STREAK_X0, STREAK_Y1 - STREAK_Y0
    scale = min(1.0, 2400 / max(cw, ch))
    lvl = slide.get_best_level_for_downsample(1 / scale) if scale < 1 else 0
    ds = slide.level_downsamples[lvl]
    crop = np.array(
        slide.read_region(
            (STREAK_X0, STREAK_Y0), lvl,
            (int(round(cw / ds)), int(round(ch / ds))),
        )
    )[:, :, :3]
    slide.close()
    s = 1.0 / ds

    COLOR = {
        "KEEP": "#2ecc71",
        "DROP pen_black": "#e67e22",
        "DROP low_tissue": "#e74c3c",
        "RESCUED": "#8e44ad",
    }

    n_show = min(12, len(streak_tiles))
    # Pick tiles that were KEPT or RESCUED but have visible black (user's concern)
    concern = df[~df.final.str.contains("pen_black")].sort_values("black_pct", ascending=False).head(n_show)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.25, wspace=0.1)
    fig.suptitle(
        f"Thick black streak audit — {sid[:16]}…\n"
        f"Orange=drop pen_black  Purple=rescued  Green=keep  |  threshold black>{PEN_THRESHOLD}%",
        fontsize=13, fontweight="bold",
    )

    ax_map = fig.add_subplot(gs[0, :])
    ax_map.imshow(crop)
    import matplotlib.patches as mpatches
    for _, r in df.iterrows():
        key = r["final"]
        if "pen_black" in key:
            c = COLOR["DROP pen_black"]
        elif "RESCUED" in key:
            c = COLOR["RESCUED"]
        elif "low_tissue" in key:
            c = COLOR["DROP low_tissue"]
        else:
            c = COLOR["KEEP"]
        rx = (r["x"] - STREAK_X0) * s
        ry = (r["y"] - STREAK_Y0) * s
        side = PATCH_SIZE * s
        ax_map.add_patch(mpatches.Rectangle(
            (rx, ry), side, side, linewidth=1.5, edgecolor=c,
            facecolor=c, alpha=0.35 if c != COLOR["KEEP"] else 0.05,
        ))
        if r["black_pct"] >= 1.0:
            ax_map.text(rx + side / 2, ry + side / 2, f"{r['black_pct']:.1f}%",
                        color="white", fontsize=6, ha="center", va="center", fontweight="bold")
    ax_map.set_title(f"Streak region x[{STREAK_X0}:{STREAK_X1}] y[{STREAK_Y0}:{STREAK_Y1}]", fontsize=11)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    # Example patches: kept/rescued despite visible black
    ncols = 6
    for i, (_, r) in enumerate(concern.iterrows()):
        if i >= ncols:
            break
        ax = fig.add_subplot(gs[1, 0] if i < 3 else gs[1, 1])
        if i >= 6:
            break
        # use subplot grid differently - 6 in bottom row
    # Simpler: 6 panels in second row using subplots
    plt.close(fig)

    fig2, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig2.suptitle(
        "Tiles NOT dropped as pen_black (but look black) — live HSV recomputed",
        fontsize=12, fontweight="bold",
    )
    show = concern.head(12)
    for ax, (_, r) in zip(axes.flat, show.itertuples()):
        rgb = read_rgb_patch(SLIDES_DIR / f"{sid}.tiff", r.x, r.y)
        ax.imshow(rgb)
        ax.set_title(
            f"({r.x},{r.y})\n{r.final}\nblack {r.black_pct:.1f}%  tissue {r.tissue_pct:.0f}%\n"
            f"cancer {r.cancer_px}px  expect={r.expected}",
            fontsize=7,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR.get("RESCUED" if "RESCUED" in r.final else "KEEP", "#333"))
            spine.set_linewidth(3)
    for ax in axes.flat[len(show):]:
        ax.axis("off")

    out1 = args.out_dir / f"blob_streak_audit_{sid[:12]}.png"
    # redo map-only figure
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 14))
    ax3.imshow(crop)
    for _, r in df.iterrows():
        key = r["final"]
        if "pen_black" in key:
            c = COLOR["DROP pen_black"]
        elif "RESCUED" in key:
            c = COLOR["RESCUED"]
        elif "low_tissue" in key:
            c = COLOR["DROP low_tissue"]
        else:
            c = COLOR["KEEP"]
        rx = (r["x"] - STREAK_X0) * s
        ry = (r["y"] - STREAK_Y0) * s
        side = PATCH_SIZE * s
        ax3.add_patch(mpatches.Rectangle(
            (rx, ry), side, side, linewidth=1.5, edgecolor=c,
            facecolor=c, alpha=0.35 if c != COLOR["KEEP"] else 0.05,
        ))
        if r["black_pct"] >= 0.5:
            ax3.text(rx + side / 2, ry + side / 2, f"{r['black_pct']:.1f}%\n{r['final'][:4]}",
                     color="yellow", fontsize=5, ha="center", va="center", fontweight="bold")
    ax3.set_title("Thick black streak — per-tile decision + black%", fontsize=11)
    ax3.axis("off")
    fig3.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    out2 = args.out_dir / f"blob_streak_kept_examples_{sid[:12]}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    csv_out = args.out_dir / f"blob_streak_audit_{sid[:12]}.csv"
    df.to_csv(csv_out, index=False)
    print(out1)
    print(out2)
    print(csv_out)


if __name__ == "__main__":
    main()
