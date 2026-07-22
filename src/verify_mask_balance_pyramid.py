"""Read-only verification: pyramid-level vs level-0 mask class balance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
MASKS = PROJECT / "data" / "masks"
OUT = PROJECT / "outputs" / "mask_balance_verification"
CLEAN = PROJECT / "data" / "radboud_clean.csv"
META = PROJECT / "data" / "train_radboud.csv"
FULL_PYR_CSV = PROJECT / "outputs" / "mask_class_balance_before_after.csv"

CLASS_NAMES = ["background", "stroma", "benign", "G3", "G4", "G5"]
CANCER = [3, 4, 5]


def pyramid_level_index(slide: openslide.OpenSlide) -> int:
    """Same as plot_mask_class_balance.py: coarsest pyramid level."""
    return slide.level_count - 1


def count_l0_chunked(mask_path: Path) -> tuple[np.ndarray, int, int]:
    s = openslide.OpenSlide(str(mask_path))
    try:
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
        return counts, w, h
    finally:
        s.close()


def count_pyramid(mask_path: Path) -> tuple[np.ndarray, int, int, int, float]:
    s = openslide.OpenSlide(str(mask_path))
    try:
        lvl = pyramid_level_index(s)
        w, h = s.level_dimensions[lvl]
        arr = np.array(s.read_region((0, 0), lvl, (w, h)))[:, :, 0]
        bc = np.bincount(arr.ravel(), minlength=6)[:6].astype(np.int64)
        ds = float(s.level_downsamples[lvl])
        return bc, lvl, w, h, ds
    finally:
        s.close()


def pct_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.zeros(6)
    return 100.0 * counts / total


def pyramid_metadata_sample(ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in ids:
        p = MASKS / f"{sid}_mask.tiff"
        s = openslide.OpenSlide(str(p.resolve()))
        try:
            lvl = pyramid_level_index(s)
            l0w, l0h = s.level_dimensions[0]
            pw, ph = s.level_dimensions[lvl]
            ds = s.level_downsamples[lvl]
            rows.append(
                {
                    "slide_id": sid,
                    "level_count": s.level_count,
                    "pyramid_level_index": lvl,
                    "l0_dims": f"{l0w}x{l0h}",
                    "pyramid_dims": f"{pw}x{ph}",
                    "downsample_factor": ds,
                    "pixels_at_pyramid": pw * ph,
                    "pixels_at_l0": l0w * l0h,
                }
            )
        finally:
            s.close()
    return pd.DataFrame(rows)


def pick_sample(clean: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common_scores = ["negative", "3+3", "3+4", "4+3", "4+4"]
    high_scores = ["4+5", "5+4", "5+5", "3+5", "5+3"]

    common_pool = clean[clean.gleason_score.isin(common_scores)]
    high_pool = clean[clean.gleason_score.isin(high_scores)]
    picked_common = common_pool.groupby("gleason_score", group_keys=False).apply(
        lambda g: g.sample(min(max(2, 10 // len(common_scores)), len(g)), random_state=seed)
    )
    if len(picked_common) < 10:
        extra = common_pool.drop(picked_common.index).sample(10 - len(picked_common), random_state=seed)
        picked_common = pd.concat([picked_common, extra])
    picked_common = picked_common.head(10)
    picked_common = picked_common.assign(category="common")

    # oversample high-grade: 2 per score where possible
    high_parts = []
    for gs in high_scores:
        g = high_pool[high_pool.gleason_score == gs]
        if len(g):
            high_parts.append(g.sample(min(2, len(g)), random_state=seed))
    picked_high = pd.concat(high_parts) if high_parts else high_pool.head(0)
    if len(picked_high) < 10:
        need = 10 - len(picked_high)
        rem = high_pool.drop(picked_high.index, errors="ignore")
        if len(rem):
            picked_high = pd.concat([picked_high, rem.sample(min(need, len(rem)), random_state=seed)])
    picked_high = picked_high.head(10).assign(category="high-grade")

    used = set(picked_common.image_id) | set(picked_high.image_id)
    rest = clean[~clean.image_id.isin(used)]
    picked_rand = rest.sample(min(10, len(rest)), random_state=seed).assign(category="random")

    return pd.concat([picked_common, picked_high, picked_rand], ignore_index=True)


def compare_slide(sid: str, category: str) -> dict:
    p = (MASKS / f"{sid}_mask.tiff").resolve()
    l0_counts, _, _ = count_l0_chunked(p)
    pyr_counts, lvl, _, _, ds = count_pyramid(p)
    l0_pct = pct_from_counts(l0_counts)
    pyr_pct = pct_from_counts(pyr_counts)
    diff = l0_pct - pyr_pct
    row = {"slide_id": sid, "category": category, "pyramid_level": lvl, "downsample_factor": ds}
    keys = ["background", "stroma", "benign", "g3", "g4", "g5"]
    for i, k in enumerate(keys):
        row[f"{k}_pct_l0"] = round(l0_pct[i], 6)
        row[f"{k}_pct_pyr"] = round(pyr_pct[i], 6)
        row[f"{k}_diff_pp"] = round(diff[i], 6)
        row[f"{k}_l0_px"] = int(l0_counts[i])
        row[f"{k}_pyr_px"] = int(pyr_counts[i])
    return row


def write_summary(
    meta_pyramid: pd.DataFrame,
    per_slide: pd.DataFrame,
    agg_l0: np.ndarray,
    agg_pyr: np.ndarray,
    vanish: list[dict],
    fabricate: list[dict],
) -> None:
    full = pd.read_csv(FULL_PYR_CSV)
    full_post = {r["class"]: r["pct_after"] for _, r in full.iterrows()}

    sample_pyr_pct = pct_from_counts(agg_pyr)
    sample_l0_pct = pct_from_counts(agg_l0)

    lines = [
        "# Mask balance pyramid verification summary",
        "",
        "## 1. What pyramid level was used?",
        "",
        "The original `plot_mask_class_balance.py` reads each mask with:",
        "",
        "```python",
        "lvl = slide.level_count - 1  # coarsest pre-baked pyramid level",
        "arr = slide.read_region((0, 0, lvl, (width, height))  # channel 0 = class ID",
        "```",
        "",
        "This is **OpenSlide `read_region` at a pre-baked TIFF pyramid level**, NOT `get_thumbnail()` and NOT a runtime PIL resize.",
        "",
        f"Across the verification sample ({len(per_slide)} slides):",
        f"- Pyramid level index: **{meta_pyramid.pyramid_level_index.mode().iloc[0]}** (mode; range {meta_pyramid.pyramid_level_index.min()}–{meta_pyramid.pyramid_level_index.max()})",
        f"- Downsample factor vs L0: **{meta_pyramid.downsample_factor.min():.0f}× to {meta_pyramid.downsample_factor.max():.0f}×** (typically 16× when level_count=3)",
        "",
        "Pre-baked pyramid pixels are produced at scan/export time by the WSI toolchain — not by our code.",
        "",
        "## 2. Largest single-slide discrepancy (percentage points)",
        "",
    ]
    for cls, key in zip(["G3", "G4", "G5"], ["g3", "g4", "g5"]):
        col = f"{key}_diff_pp"
        abs_diff = per_slide[col].abs()
        idx = abs_diff.idxmax()
        r = per_slide.loc[idx]
        lines.append(
            f"- **{cls}**: max |diff| = {abs_diff.max():.4f} pp on `{r.slide_id}` "
            f"(L0 {r[f'{key}_pct_l0']:.4f}% vs pyramid {r[f'{key}_pct_pyr']:.4f}%)"
        )

    lines += ["", "## 3. Systematic bias (30-slide sample)", ""]
    for cls, key in zip(CLASS_NAMES, ["background", "stroma", "benign", "g3", "g4", "g5"]):
        col = f"{key}_diff_pp"
        mean_d = per_slide[col].mean()
        med_d = per_slide[col].median()
        lines.append(f"- **{cls}**: mean diff (L0−pyr) = {mean_d:+.4f} pp, median = {med_d:+.4f} pp")

    lines += ["", "## 4. Known failure patterns in sample", ""]
    lines.append(f"### Vanishing (L0>0, pyramid=0 pixels)")
    if vanish:
        for v in vanish:
            lines.append(f"- `{v['slide_id']}` class {v['class']}: L0={v['l0_px']} px, pyramid=0")
    else:
        lines.append("- **None** in the 30-slide sample.")

    lines.append("")
    lines.append("### Fabricated (L0=0, pyramid>0 pixels)")
    if fabricate:
        for v in fabricate:
            lines.append(f"- `{v['slide_id']}` class {v['class']}: L0=0, pyramid={v['pyr_px']} px")
    else:
        lines.append("- **None** in the 30-slide sample.")

    lines += ["", "## 5. Aggregate comparison (30-slide sample)", ""]
    lines.append("| Class | 30-slide L0 % | 30-slide pyramid % | diff (pp) | Full-dataset pyramid % (post-QC) |")
    lines.append("|---|---:|---:|---:|---:|")
    for i, cls in enumerate(CLASS_NAMES):
        key = ["background", "stroma", "benign", "g3", "g4", "g5"][i]
        d = sample_l0_pct[i] - sample_pyr_pct[i]
        lines.append(
            f"| {cls} | {sample_l0_pct[i]:.4f} | {sample_pyr_pct[i]:.4f} | {d:+.4f} | {full_post.get(cls, float('nan')):.4f} |"
        )

    cancer_d = sample_l0_pct[3:6].sum() - sample_pyr_pct[3:6].sum()
    lines += [
        "",
        f"**30-slide test:** L0 vs pyramid aggregate differs by at most "
        f"{max(abs(sample_l0_pct[i]-sample_pyr_pct[i]) for i in range(6)):.4f} pp on any single class; "
        f"combined cancer grades (G3+G4+G5) diff = {cancer_d:+.4f} pp.",
        "",
        "## 6. Recommendation",
        "",
    ]

    max_cancer_diff = max(
        abs(per_slide["g3_diff_pp"].max()),
        abs(per_slide["g4_diff_pp"].max()),
        abs(per_slide["g5_diff_pp"].max()),
    )
    if not vanish and not fabricate and float(np.max(np.abs(sample_l0_pct - sample_pyr_pct))) < 0.05:
        lines.append(
            "The pyramid-level dataset-wide percentages look **trustworthy for reporting coarse balance** "
            "(background/stroma/cancer split). Per-slide rare-grade percentages can differ more, but aggregate "
            "bias across the 30-slide sample is sub-0.05 pp per class."
        )
        lines.append("")
        lines.append(
            "A full level-0 Slurm job is **not required** before using these numbers for high-level class-balance "
            "discussion. Run L0 aggregate only if you need exact cancer-grade percentages beyond ~0.01 pp precision."
        )
    else:
        lines.append(
            "Caution warranted: discrepancies or failure-pattern slides were found. See per-slide CSV and consider "
            "a larger L0 Slurm verification before treating cancer-grade percentages as final."
        )

    (OUT / "mask_balance_verification_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(CLEAN)
    meta = pd.read_csv(META)[["image_id", "gleason_score"]]
    if "gleason_score" in clean.columns:
        clean = clean.drop(columns=["gleason_score"], errors="ignore")
    clean = clean.merge(meta, on="image_id", how="left")
    sample = pick_sample(clean)
    sample[["image_id", "gleason_score", "category"]].to_csv(OUT / "verification_sample_slides.csv", index=False)

    meta = pyramid_metadata_sample(sample.image_id.tolist())
    meta.to_csv(OUT / "pyramid_level_metadata_sample.csv", index=False)

    rows = []
    agg_l0 = np.zeros(6, dtype=np.int64)
    agg_pyr = np.zeros(6, dtype=np.int64)
    vanish: list[dict] = []
    fabricate: list[dict] = []

    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="L0 vs pyramid"):
        row = compare_slide(r.image_id, r.category)
        rows.append(row)
        for i in range(6):
            agg_l0[i] += row[f"{['background','stroma','benign','g3','g4','g5'][i]}_l0_px"]
            agg_pyr[i] += row[f"{['background','stroma','benign','g3','g4','g5'][i]}_pyr_px"]
        for g, name in zip(CANCER, ["G3", "G4", "G5"]):
            k = name.lower()
            l0p, pyrp = row[f"{k}_l0_px"], row[f"{k}_pyr_px"]
            if l0p > 0 and pyrp == 0:
                vanish.append({"slide_id": r.image_id, "class": name, "l0_px": l0p})
            if l0p == 0 and pyrp > 0:
                fabricate.append({"slide_id": r.image_id, "class": name, "pyr_px": pyrp})

    per_slide = pd.DataFrame(rows)
    out_cols = ["slide_id", "category"]
    for k in ["background", "stroma", "benign", "g3", "g4", "g5"]:
        out_cols += [f"{k}_pct_l0", f"{k}_pct_pyr", f"{k}_diff_pp"]
    per_slide[out_cols].to_csv(OUT / "level0_vs_pyramid_per_slide.csv", index=False)
    per_slide.to_csv(OUT / "level0_vs_pyramid_per_slide_full.csv", index=False)

    bias = []
    for k, label in zip(
        ["background", "stroma", "benign", "g3", "g4", "g5"],
        CLASS_NAMES,
    ):
        col = f"{k}_diff_pp"
        bias.append(
            {
                "class": label,
                "mean_diff_pp_l0_minus_pyr": per_slide[col].mean(),
                "median_diff_pp_l0_minus_pyr": per_slide[col].median(),
                "std_diff_pp": per_slide[col].std(),
            }
        )
    pd.DataFrame(bias).to_csv(OUT / "bias_summary.csv", index=False)

    json.dump(
        {"vanishing": vanish, "fabricated": fabricate},
        (OUT / "failure_pattern_slides.json").open("w"),
        indent=2,
    )

    write_summary(meta, per_slide, agg_l0, agg_pyr, vanish, fabricate)
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
