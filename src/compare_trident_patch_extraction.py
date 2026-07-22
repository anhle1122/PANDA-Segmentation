"""Compare patch extraction: naive grid (Pipeline A) vs Trident-Otsu (Pipeline B).

Read-only diagnostic — does not modify extract_patches.py or radboud_clean.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "vendor" / "TRIDENT"))
sys.path.insert(0, str(PROJECT / "src"))

from patch_utils import MASKS_DIR, PATCH_SIZE, SLIDES_DIR  # noqa: E402

OUT = PROJECT / "outputs" / "trident_patch_comparison"
CLEAN = PROJECT / "data" / "radboud_clean.csv"
META = PROJECT / "data" / "train_radboud.csv"

TISSUE_FRAC_MIN = 0.50
CANCER_LABELS = (3, 4, 5)
TRIDENT_MAG = 20
TRIDENT_OVERLAP = 0
TRIDENT_MIN_TISSUE_PROP = 0.0


@dataclass
class PatchDecision:
    x: int
    y: int
    tissue_frac: float
    has_cancer: bool
    kept: bool
    kept_via: str = ""
    discard_reason: str = ""


@dataclass
class SlideResult:
    slide_id: str
    candidate_patches_total: int = 0
    kept_total: int = 0
    discarded_total: int = 0
    discarded_zero_tissue: int = 0
    discarded_under_50pct_no_cancer: int = 0
    kept_via_50pct_rule: int = 0
    kept_via_cancer_rescue: int = 0
    runtime_sec: float = 0.0
    trident_candidate_patches_total: int | None = None
    trident_tissue_contour_area_px: int | None = None
    decisions: list[PatchDecision] = field(default_factory=list)
    kept_coords: set[tuple[int, int]] = field(default_factory=set)
    candidate_coords: set[tuple[int, int]] = field(default_factory=set)


def pick_sample(clean: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    common_scores = ["negative", "3+3", "4+3"]
    high_scores = ["4+5", "5+4", "5+5"]

    common_pool = clean[clean.gleason_score.isin(common_scores)]
    high_pool = clean[clean.gleason_score.isin(high_scores)]
    picked_common = common_pool.groupby("gleason_score", group_keys=False).apply(
        lambda g: g.sample(1, random_state=seed)
    )
    if len(picked_common) < 5:
        extra = common_pool.drop(picked_common.index).sample(5 - len(picked_common), random_state=seed)
        picked_common = pd.concat([picked_common, extra])
    picked_common = picked_common.head(5).assign(category="common")

    high_parts = []
    for gs in high_scores:
        g = high_pool[high_pool.gleason_score == gs]
        if len(g):
            high_parts.append(g.sample(min(2, len(g)), random_state=seed))
    picked_high = pd.concat(high_parts) if high_parts else high_pool.head(0)
    if len(picked_high) < 5:
        rem = high_pool.drop(picked_high.index, errors="ignore")
        if len(rem):
            picked_high = pd.concat(
                [picked_high, rem.sample(min(5 - len(picked_high), len(rem)), random_state=seed)]
            )
    picked_high = picked_high.head(5).assign(category="high-grade")

    used = set(picked_common.image_id) | set(picked_high.image_id)
    rest = clean[~clean.image_id.isin(used)]
    picked_rand = rest.sample(min(5, len(rest)), random_state=seed).assign(category="random")
    return pd.concat([picked_common, picked_high, picked_rand], ignore_index=True)


def read_mask_patch(mask_slide: openslide.OpenSlide, x: int, y: int, size: int) -> np.ndarray:
    arr = np.array(mask_slide.read_region((x, y), 0, (size, size)))
    return arr[:, :, 0] if arr.ndim == 3 else arr


def classify_labels(labels: np.ndarray) -> tuple[float, bool]:
    area = labels.size
    tissue = int((labels > 0).sum())
    has_cancer = bool(any(int((labels == g).sum()) > 0 for g in CANCER_LABELS))
    return (tissue / area if area else 0.0), has_cancer


def decision_from_labels(x: int, y: int, labels: np.ndarray) -> PatchDecision:
    tissue_frac, has_cancer = classify_labels(labels)
    if tissue_frac == 0.0:
        return PatchDecision(x, y, tissue_frac, has_cancer, False, discard_reason="zero_tissue")
    if tissue_frac >= TISSUE_FRAC_MIN:
        return PatchDecision(x, y, tissue_frac, has_cancer, True, kept_via="50pct")
    if has_cancer:
        return PatchDecision(x, y, tissue_frac, has_cancer, True, kept_via="cancer_rescue")
    return PatchDecision(x, y, tissue_frac, has_cancer, False, discard_reason="under_50_no_cancer")


def summarize_decisions(decisions: list[PatchDecision]) -> SlideResult:
    r = SlideResult(slide_id="")
    r.candidate_patches_total = len(decisions)
    r.candidate_coords = {(d.x, d.y) for d in decisions}
    for d in decisions:
        if d.kept:
            r.kept_total += 1
            r.kept_coords.add((d.x, d.y))
            if d.kept_via == "50pct":
                r.kept_via_50pct_rule += 1
            elif d.kept_via == "cancer_rescue":
                r.kept_via_cancer_rescue += 1
        else:
            r.discarded_total += 1
            if d.discard_reason == "zero_tissue":
                r.discarded_zero_tissue += 1
            elif d.discard_reason == "under_50_no_cancer":
                r.discarded_under_50pct_no_cancer += 1
    r.decisions = decisions
    return r


def grid_coords(width: int, height: int, patch_size: int) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(0, height - patch_size + 1, patch_size)
        for x in range(0, width - patch_size + 1, patch_size)
    ]


def run_pipeline_a(slide_id: str) -> SlideResult:
    t0 = time.perf_counter()
    mask_slide = openslide.OpenSlide(str(MASKS_DIR / f"{slide_id}_mask.tiff"))
    try:
        width, height = mask_slide.dimensions
        decisions = [
            decision_from_labels(x, y, read_mask_patch(mask_slide, x, y, PATCH_SIZE))
            for x, y in grid_coords(width, height, PATCH_SIZE)
        ]
    finally:
        mask_slide.close()
    result = summarize_decisions(decisions)
    result.slide_id = slide_id
    result.runtime_sec = time.perf_counter() - t0
    return result


def trident_tissue_area_px(gdf) -> int:
    if gdf is None or len(gdf) == 0:
        return 0
    try:
        return int(round(gdf.union_all().area))
    except Exception:
        return int(round(gdf.geometry.area.sum()))


def run_pipeline_b(slide_id: str, job_root: Path) -> SlideResult:
    from trident import load_wsi
    from trident.IO import read_coords
    from trident.segmentation_models import segmentation_model_factory

    t0 = time.perf_counter()
    slide_path = SLIDES_DIR / f"{slide_id}.tiff"
    mask_path = MASKS_DIR / f"{slide_id}_mask.tiff"
    job_dir = job_root / slide_id
    job_dir.mkdir(parents=True, exist_ok=True)

    patch_size_l0 = PATCH_SIZE
    trident_coords: list[tuple[int, int]] = []
    tissue_area = 0

    with load_wsi(str(slide_path), reader_type="openslide") as wsi:
        seg_model = segmentation_model_factory("otsu", confidence_thresh=0.5)
        wsi.segment_tissue(
            segmentation_model=seg_model,
            target_mag=seg_model.target_mag,
            job_dir=str(job_dir),
            device="cpu",
            holes_are_tissue=True,
        )
        tissue_area = trident_tissue_area_px(getattr(wsi, "gdf_contours", None))
        mag_str = f"{float(TRIDENT_MAG):g}"
        save_coords = str(job_dir / f"{mag_str}x_{PATCH_SIZE}px_{TRIDENT_OVERLAP}px_overlap")
        coords_path = wsi.extract_tissue_coords(
            target_mag=TRIDENT_MAG,
            patch_size=PATCH_SIZE,
            save_coords=save_coords,
            overlap=TRIDENT_OVERLAP,
            min_tissue_proportion=TRIDENT_MIN_TISSUE_PROP,
        )
        attrs, coord_arr = read_coords(coords_path)
        l0_mag = float(attrs.get("level0_magnification", TRIDENT_MAG))
        tgt_mag = float(attrs.get("target_magnification", TRIDENT_MAG))
        patch_size_l0 = int(round(PATCH_SIZE * l0_mag / tgt_mag))
        trident_coords = [(int(x), int(y)) for x, y in coord_arr]

    mask_slide = openslide.OpenSlide(str(mask_path))
    try:
        decisions = [
            decision_from_labels(x, y, read_mask_patch(mask_slide, x, y, patch_size_l0))
            for x, y in trident_coords
        ]
    finally:
        mask_slide.close()

    result = summarize_decisions(decisions)
    result.slide_id = slide_id
    result.trident_candidate_patches_total = len(trident_coords)
    result.trident_tissue_contour_area_px = tissue_area
    result.runtime_sec = time.perf_counter() - t0
    return result


def slide_row_a(r: SlideResult) -> dict:
    return {
        "slide_id": r.slide_id,
        "candidate_patches_total": r.candidate_patches_total,
        "kept_total": r.kept_total,
        "discarded_total": r.discarded_total,
        "discarded_zero_tissue": r.discarded_zero_tissue,
        "discarded_under_50pct_no_cancer": r.discarded_under_50pct_no_cancer,
        "kept_via_50pct_rule": r.kept_via_50pct_rule,
        "kept_via_cancer_rescue": r.kept_via_cancer_rescue,
        "runtime_sec": round(r.runtime_sec, 2),
    }


def slide_row_b(r: SlideResult) -> dict:
    row = slide_row_a(r)
    row["trident_candidate_patches_total"] = r.trident_candidate_patches_total
    row["trident_tissue_contour_area_px"] = r.trident_tissue_contour_area_px
    return row


def compare_slides(a: SlideResult, b: SlideResult) -> dict:
    pct_a = 100.0 * a.kept_total / a.candidate_patches_total if a.candidate_patches_total else 0.0
    pct_b = 100.0 * b.kept_total / b.candidate_patches_total if b.candidate_patches_total else 0.0
    return {
        "slide_id": a.slide_id,
        "A_candidate_patches": a.candidate_patches_total,
        "B_candidate_patches": b.trident_candidate_patches_total,
        "candidate_diff_B_minus_A": (b.trident_candidate_patches_total or 0) - a.candidate_patches_total,
        "A_kept": a.kept_total,
        "B_kept": b.kept_total,
        "kept_diff_B_minus_A": b.kept_total - a.kept_total,
        "A_pct_kept": round(pct_a, 4),
        "B_pct_kept": round(pct_b, 4),
        "A_discarded_under50_no_cancer": a.discarded_under_50pct_no_cancer,
        "B_discarded_under50_no_cancer": b.discarded_under_50pct_no_cancer,
        "A_cancer_rescue_kept": a.kept_via_cancer_rescue,
        "B_cancer_rescue_kept": b.kept_via_cancer_rescue,
        "A_runtime_sec": round(a.runtime_sec, 2),
        "B_runtime_sec": round(b.runtime_sec, 2),
        "trident_tissue_contour_area_px": b.trident_tissue_contour_area_px,
    }


def find_a_only_b_only(a: SlideResult, b: SlideResult) -> tuple[list[dict], list[dict]]:
    a_only, b_only = [], []
    for d in a.decisions:
        if d.kept and (d.x, d.y) not in b.candidate_coords:
            a_only.append(
                {
                    "slide_id": a.slide_id,
                    "x": d.x,
                    "y": d.y,
                    "category": "cancer_rescue_lost" if d.kept_via == "cancer_rescue" else "other_kept_lost",
                    "tissue_frac": round(d.tissue_frac, 4),
                    "kept_via": d.kept_via,
                    "pipeline": "A_only_not_in_B_candidates",
                }
            )
    for d in b.decisions:
        if d.kept and (d.x, d.y) not in a.kept_coords:
            b_only.append(
                {
                    "slide_id": b.slide_id,
                    "x": d.x,
                    "y": d.y,
                    "category": d.kept_via,
                    "tissue_frac": round(d.tissue_frac, 4),
                    "kept_via": d.kept_via,
                    "pipeline": "B_only_not_kept_in_A",
                }
            )
    return a_only, b_only


def read_mask_thumbnail_nn(mask_path: Path, max_edge: int = 2048) -> np.ndarray:
    slide = openslide.OpenSlide(str(mask_path))
    try:
        w, h = slide.dimensions
        scale = min(max_edge / w, max_edge / h, 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        lvl = slide.get_best_level_for_downsample(1 / scale)
        ds = slide.level_downsamples[lvl]
        arr = np.array(slide.read_region((0, 0), lvl, (int(tw * scale / ds), int(th * scale / ds))))[:, :, 0]
        return np.array(Image.fromarray(arr).resize((tw, th), resample=Image.NEAREST))
    finally:
        slide.close()


def read_wsi_thumbnail(slide_path: Path, max_edge: int = 2048) -> np.ndarray:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        w, h = slide.dimensions
        scale = max_edge / max(w, h)
        return np.array(slide.get_thumbnail((max(1, int(w * scale)), max(1, int(h * scale)))))[:, :, :3]
    finally:
        slide.close()


def visualize_a_only_cases(rows: list[dict], out_dir: Path, max_viz: int = 20) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    priority = [r for r in rows if r["category"] == "cancer_rescue_lost"]
    other = [r for r in rows if r["category"] != "cancer_rescue_lost"]
    for row in (priority + other)[:max_viz]:
        sid = row["slide_id"]
        wsi_thumb = read_wsi_thumbnail(SLIDES_DIR / f"{sid}.tiff")
        mask_thumb = read_mask_thumbnail_nn(MASKS_DIR / f"{sid}_mask.tiff")
        cmap = np.zeros((*mask_thumb.shape, 3), dtype=np.uint8)
        for cls, col in {1: (0, 0, 255), 2: (0, 180, 0), 3: (255, 255, 0), 4: (255, 140, 0), 5: (255, 0, 0)}.items():
            cmap[mask_thumb == cls] = col
        slide = openslide.OpenSlide(str(MASKS_DIR / f"{sid}_mask.tiff"))
        try:
            w, _ = slide.dimensions
        finally:
            slide.close()
        scale = wsi_thumb.shape[1] / w
        x, y = row["x"], row["y"]
        px, py, ps = x * scale, y * scale, PATCH_SIZE * scale
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(axes, [wsi_thumb, cmap, wsi_thumb], ["WSI", "Mask (NN)", "Overlay"]):
            ax.imshow(img if title != "Overlay" else wsi_thumb)
            if title == "Overlay":
                ax.imshow(cmap, alpha=0.45)
            ax.add_patch(mpatches.Rectangle((px, py), ps, ps, fill=False, edgecolor="lime", linewidth=2))
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{sid[:12]}... {row['category']} x={x} y={y} tissue={row['tissue_frac']:.1%}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"a_only_{sid[:12]}_x{x}_y{y}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def write_summary(comp: pd.DataFrame, a_only: list[dict], b_only: list[dict]) -> None:
    agg = {
        "A_candidates": int(comp["A_candidate_patches"].sum()),
        "B_candidates": int(comp["B_candidate_patches"].sum()),
        "A_kept": int(comp["A_kept"].sum()),
        "B_kept": int(comp["B_kept"].sum()),
        "A_runtime": float(comp["A_runtime_sec"].sum()),
        "B_runtime": float(comp["B_runtime_sec"].sum()),
        "A_rescue": int(comp["A_cancer_rescue_kept"].sum()),
        "B_rescue": int(comp["B_cancer_rescue_kept"].sum()),
    }
    rescue_lost = [r for r in a_only if r["category"] == "cancer_rescue_lost"]
    cand_red = 100.0 * (1 - agg["B_candidates"] / agg["A_candidates"]) if agg["A_candidates"] else 0.0
    kept_diff = agg["B_kept"] - agg["A_kept"]
    lines = [
        "# Trident-Otsu vs naive-grid patch extraction comparison",
        "",
        f"Keep rule: tissue >= {TISSUE_FRAC_MIN:.0%} OR any G3/G4/G5 pixel. Patch size {PATCH_SIZE} @ L0.",
        "",
        "## Aggregate (15 slides)",
        f"- Candidates: A={agg['A_candidates']:,}, B={agg['B_candidates']:,} ({cand_red:.1f}% reduction)",
        f"- Kept: A={agg['A_kept']:,}, B={agg['B_kept']:,} (diff {kept_diff:+d})",
        f"- Cancer-rescue kept: A={agg['A_rescue']}, B={agg['B_rescue']}",
        f"- Runtime: A={agg['A_runtime']:.0f}s, B={agg['B_runtime']:.0f}s",
        "",
        f"- A-only (kept in A, outside B candidates): {len(a_only)} ({len(rescue_lost)} cancer-rescue)",
        f"- B-only (kept in B, not in A kept): {len(b_only)}",
        "",
        "See `pipeline_comparison_per_slide.csv` and `a_only_or_b_only_summary.csv`.",
    ]
    (OUT / "trident_comparison_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-viz", type=int, default=20)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(CLEAN)
    meta = pd.read_csv(META)[["image_id", "gleason_score"]]
    if "gleason_score" in clean.columns:
        clean = clean.drop(columns=["gleason_score"], errors="ignore")
    clean = clean.merge(meta, on="image_id", how="left")
    sample = pick_sample(clean, seed=args.seed)
    sample.to_csv(OUT / "comparison_sample_slides.csv", index=False)

    results_a = [run_pipeline_a(sid) for sid in tqdm(sample.image_id, desc="Pipeline A")]
    results_b = [run_pipeline_b(sid, OUT / "trident_jobs") for sid in tqdm(sample.image_id, desc="Pipeline B")]

    pd.DataFrame([slide_row_a(r) for r in results_a]).to_csv(OUT / "pipeline_A_no_trident_results.csv", index=False)
    pd.DataFrame([slide_row_b(r) for r in results_b]).to_csv(OUT / "pipeline_B_with_trident_results.csv", index=False)
    comp = pd.DataFrame([compare_slides(a, b) for a, b in zip(results_a, results_b)])
    comp.to_csv(OUT / "pipeline_comparison_per_slide.csv", index=False)

    a_only_all, b_only_all = [], []
    for a, b in zip(results_a, results_b):
        ao, bo = find_a_only_b_only(a, b)
        a_only_all.extend(ao)
        b_only_all.extend(bo)
    pd.DataFrame(a_only_all + b_only_all).to_csv(OUT / "a_only_or_b_only_summary.csv", index=False)
    if a_only_all:
        visualize_a_only_cases(a_only_all, OUT / "category_a_only_patches", args.max_viz)
    write_summary(comp, a_only_all, b_only_all)
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
