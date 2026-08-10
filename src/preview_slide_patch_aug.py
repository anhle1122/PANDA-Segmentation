#!/usr/bin/env python3
"""Preview slide-level vs patch-level augmentation — tissue-rich patches only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

from patch_utils import OUTPUTS, PROJECT
from train.augmentations import (
    apply_image_only,
    apply_slide_consistent,
    build_train_augmentor,
    sample_slide_aug_params,
)

SLIDES = Path("/common/omarmlab/members/anh/panda_data/slides")
PEN = PROJECT / "outputs" / "pen_filter_v33"
SPLITS = PROJECT / "outputs" / "splits"
OUT = OUTPUTS / "augmentation_examples" / "slide_and_patch_preview"


def slide_path(sid: str) -> Path:
    paths = list(SLIDES.glob(f"{sid[:12]}*.tiff"))
    if not paths:
        raise FileNotFoundError(sid)
    return paths[0]


def thumb(sid: str, box=(320, 560)) -> np.ndarray:
    sl = openslide.OpenSlide(str(slide_path(sid)))
    im = sl.get_thumbnail(box).convert("RGB")
    sl.close()
    arr = np.asarray(im)
    canvas = np.full((box[1], box[0], 3), 245, dtype=np.uint8)
    y0 = (box[1] - arr.shape[0]) // 2
    x0 = (box[0] - arr.shape[1]) // 2
    canvas[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]] = arr
    return canvas


def read_patch(sid: str, x: int, y: int, size: int = 256) -> np.ndarray:
    sl = openslide.OpenSlide(str(slide_path(sid)))
    im = sl.read_region((x, y), 0, (size, size)).convert("RGB")
    sl.close()
    return np.asarray(im)


def tissue_score(rgb: np.ndarray) -> float:
    """Fraction of non-glass pixels (not near-white)."""
    # glass ≈ high RGB; tissue darker / chromatic
    white = (rgb[..., 0] > 220) & (rgb[..., 1] > 220) & (rgb[..., 2] > 220)
    return float((~white).mean())


def pick_tissue_patches(sid: str, tr: pd.DataFrame, k: int = 3) -> list[tuple[int, int, np.ndarray]]:
    """Prefer v33 keep patches with high tissue_pct; fall back to RGB tissue score."""
    rows = tr[tr.image_id.astype(str) == sid][["x", "y"]].copy()
    pen = PEN / f"pen_filter_v33_{sid[:12]}.csv"
    scored: list[tuple[float, int, int]] = []
    if pen.exists():
        pdf = pd.read_csv(pen)
        m = rows.merge(pdf, on=["x", "y"], how="left")
        keep = m[m.get("v33_action", pd.Series(dtype=str)) == "keep"] if "v33_action" in m.columns else m
        if len(keep) == 0:
            keep = m
        if "tissue_pct" in keep.columns:
            keep = keep.sort_values("tissue_pct", ascending=False)
            for r in keep.head(40).itertuples():
                scored.append((float(r.tissue_pct), int(r.x), int(r.y)))
    if not scored:
        # sample and score by RGB
        sample = rows.sample(n=min(40, len(rows)), random_state=0)
        for r in sample.itertuples():
            rgb = read_patch(sid, int(r.x), int(r.y))
            scored.append((tissue_score(rgb) * 100.0, int(r.x), int(r.y)))
        scored.sort(reverse=True)

    out = []
    for score, x, y in scored:
        rgb = read_patch(sid, x, y)
        ts = tissue_score(rgb)
        if ts < 0.35:  # skip glass
            continue
        out.append((x, y, rgb))
        if len(out) >= k:
            break
    if len(out) < k:
        # last resort: best scored even if glassy
        for score, x, y in scored:
            if any(x == a and y == b for a, b, _ in out):
                continue
            out.append((x, y, read_patch(sid, x, y)))
            if len(out) >= k:
                break
    return out


def pick_slides(tr: pd.DataFrame, n: int = 2) -> list[str]:
    """Slides with many high-tissue keep patches."""
    cands = []
    for sid, n_patches in tr.groupby("image_id").size().sort_values(ascending=False).head(80).items():
        sid = str(sid)
        pen = PEN / f"pen_filter_v33_{sid[:12]}.csv"
        if not pen.exists():
            continue
        pdf = pd.read_csv(pen)
        keep = pdf[pdf.v33_action == "keep"] if "v33_action" in pdf.columns else pdf
        if "tissue_pct" not in keep.columns or len(keep) < 8:
            continue
        top = float(keep.tissue_pct.nlargest(5).mean())
        if top < 40:
            continue
        cands.append((top, int(n_patches), sid))
    cands.sort(reverse=True)
    return [s for _, _, s in cands[:n]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260809)
    tr = pd.read_csv(SPLITS / "panda_train.csv")
    sids = pick_slides(tr, n=2)
    if len(sids) < 2:
        raise SystemExit("Could not find 2 tissue-rich slides")
    print("slides", [s[:12] for s in sids])
    patch_aug = build_train_augmentor()

    # Combo: 2 slides × (slide orig | slide aug | 3 tissue patches before/after)
    for si, sid in enumerate(sids):
        params = sample_slide_aug_params(rng)
        # force some visible geometry/color for demo (still realistic ranges)
        # keep sampled params — already random

        t0 = thumb(sid)
        t1, _, _ = apply_slide_consistent(t0.copy(), None, None, params)
        patches = pick_tissue_patches(sid, tr, k=3)
        print(sid[:12], "patches", [(x, y, round(tissue_score(rgb), 2)) for x, y, rgb in patches])

        n_p = len(patches)
        fig, axes = plt.subplots(2, 2 + n_p, figsize=(4 * (2 + n_p), 9))
        axes[0, 0].imshow(t0)
        axes[0, 0].set_title(f"SLIDE original\n{sid[:12]}…", fontsize=11)
        axes[0, 0].axis("off")
        axes[0, 1].imshow(t1)
        axes[0, 1].set_title(
            f"SLIDE-level aug\ndh={params.dh:.3f} de={params.de:.3f}\n"
            f"hflip={params.hflip} vflip={params.vflip} rot90={params.rotate90}",
            fontsize=10,
        )
        axes[0, 1].axis("off")
        for j in range(n_p):
            axes[0, 2 + j].imshow(patches[j][2])
            axes[0, 2 + j].set_title(
                f"PATCH original\n({patches[j][0]},{patches[j][1]})\ntissue={tissue_score(patches[j][2]):.0%}",
                fontsize=10,
            )
            axes[0, 2 + j].axis("off")

        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[1, 0].text(
            0.1,
            0.5,
            "Bottom row =\nsame patches after\nslide-level then\npatch-level aug",
            transform=axes[1, 0].transAxes,
            fontsize=12,
            va="center",
        )
        for j, (x, y, rgb) in enumerate(patches):
            p_slide, _, _ = apply_slide_consistent(rgb.copy(), None, None, params)
            p_both = apply_image_only(patch_aug, p_slide.copy())
            axes[1, 2 + j].imshow(p_both)
            axes[1, 2 + j].set_title("after slide+patch aug", fontsize=10)
            axes[1, 2 + j].axis("off")
        # hide unused bottom cells if any
        for j in range(2):
            axes[1, j].axis("off")

        fig.suptitle(f"Tissue-rich aug preview — slide {si+1}: {sid[:16]}…", fontsize=13)
        fig.tight_layout()
        outp = OUT / f"slide_{si+1}_{sid[:12]}_tissue_before_after.png"
        fig.savefig(outp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", outp)

    # side-by-side 2-slide overview (thumbs + one best patch each)
    fig, axes = plt.subplots(2, 4, figsize=(14, 10))
    for row, sid in enumerate(sids):
        params = sample_slide_aug_params(np.random.default_rng(100 + row))
        t0 = thumb(sid)
        t1, _, _ = apply_slide_consistent(t0.copy(), None, None, params)
        x, y, p0 = pick_tissue_patches(sid, tr, k=1)[0]
        p1, _, _ = apply_slide_consistent(p0.copy(), None, None, params)
        p2 = apply_image_only(patch_aug, p1.copy())
        axes[row, 0].imshow(t0); axes[row, 0].set_title(f"{sid[:12]} slide orig"); axes[row, 0].axis("off")
        axes[row, 1].imshow(t1); axes[row, 1].set_title("slide-level aug"); axes[row, 1].axis("off")
        axes[row, 2].imshow(p0); axes[row, 2].set_title(f"tissue patch orig ({tissue_score(p0):.0%})"); axes[row, 2].axis("off")
        axes[row, 3].imshow(p2); axes[row, 3].set_title("slide+patch aug"); axes[row, 3].axis("off")
    fig.suptitle("2 slides — tissue patches only (no glass)", fontsize=14)
    fig.tight_layout()
    combo = OUT / "two_slides_tissue_before_after.png"
    fig.savefig(combo, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", combo)


if __name__ == "__main__":
    main()
