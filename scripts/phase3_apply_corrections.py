#!/usr/bin/env python3
"""Phase 3 Steps 1+3–5: rebuild diagnostic at min_area_pct=0.03 and apply corrections.

STOP after writing corrected labels + printing summary (no retrain).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from isup_diagnostic import derive_grade
from patch_utils import PROJECT

REPORT_V1 = PROJECT / "outputs/pseudo_label/diagnostic_report.csv"
REPORT_V2 = PROJECT / "outputs/pseudo_label/diagnostic_report_v2.csv"
REPORT_V2_SUMMARY = PROJECT / "outputs/pseudo_label/diagnostic_report_v2_summary.json"
CORRECTED_DIR = PROJECT / "outputs/pseudo_label/corrected_labels"
MANIFEST_CSV = PROJECT / "outputs/pseudo_label/correction_manifest.csv"
SUMMARY_JSON = PROJECT / "outputs/pseudo_label/correction_summary.json"
H5_DIR = PROJECT / "outputs/kept_extract/raw"
TRAIN_SPLIT = PROJECT / "outputs/splits/panda_train.csv"
# Diagnostic used for corrections must stay at 5% (v1). Do not apply 3% v2.
MIN_AREA_PCT = 0.05
NUM_CLASSES = 6


def gleason_to_classes(metadata_gleason: str) -> set[int]:
    g = str(metadata_gleason).strip().lower()
    if g in {"negative", "benign", "0+0", "nan", ""}:
        return set()
    if "+" not in g:
        raise ValueError(f"Unexpected gleason string: {metadata_gleason!r}")
    a, b = g.split("+", 1)
    return {int(a), int(b)}


def primary_gleason_class(metadata_gleason: str) -> int:
    return int(str(metadata_gleason).strip().split("+", 1)[0])


def rebuild_diagnostic_v2() -> pd.DataFrame:
    """Re-derive ISUP from stored pred pixel counts at min_area_pct=0.03."""
    df = pd.read_csv(REPORT_V1)
    rows = []
    for _, r in df.iterrows():
        counts = np.array(
            [r[f"pred_pixels_{c}"] for c in range(NUM_CLASSES)], dtype=np.float64
        )
        derived_gleason, derived_isup = derive_grade(counts, min_area_pct=MIN_AREA_PCT)
        cancer = float(counts[3:6].sum())
        tissue = float(counts[1:6].sum())
        cancer_area_pct = float(100.0 * cancer / tissue) if tissue > 0 else 0.0
        meta_isup = int(r.metadata_isup)
        rows.append(
            {
                "slide_id": r.slide_id,
                "metadata_gleason": r.metadata_gleason,
                "metadata_isup": meta_isup,
                "derived_gleason": derived_gleason,
                "derived_isup": int(derived_isup),
                "match": bool(int(derived_isup) == meta_isup),
                "n_patches": int(r.n_patches),
                **{f"pred_pixels_{c}": int(counts[c]) for c in range(NUM_CLASSES)},
                "cancer_frac_g3": float(r.cancer_frac_g3),
                "cancer_frac_g4": float(r.cancer_frac_g4),
                "cancer_frac_g5": float(r.cancer_frac_g5),
                "cancer_area_pct": cancer_area_pct,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_V2, index=False)

    by = {}
    for g in range(6):
        sub = out[out.metadata_isup == g]
        by[str(g)] = {
            "count": int(len(sub)),
            "matches": int(sub.match.sum()),
            "match_rate": float(sub.match.mean()) if len(sub) else 0.0,
            "mismatches": int((~sub.match).sum()),
        }
    summary = {
        "source": str(REPORT_V1),
        "min_area_pct": MIN_AREA_PCT,
        "note": (
            "Re-derived from stored pred_pixels (no re-inference); "
            "identical to rerunning Phase 2 at min_area_pct=0.03"
        ),
        "checkpoint": str(
            PROJECT
            / "outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth"
        ),
        "total_slides": int(len(out)),
        "matches": int(out.match.sum()),
        "mismatches": int((~out.match).sum()),
        "match_rate": float(out.match.mean()),
        "isup0_mismatches": int(((out.metadata_isup == 0) & (~out.match)).sum()),
        "isup0_mismatches_v1_at_0p05": 623,
        "by_metadata_isup": by,
    }
    REPORT_V2_SUMMARY.write_text(json.dumps(summary, indent=2))
    print("Wrote", REPORT_V2)
    print(json.dumps({k: summary[k] for k in (
        "match_rate", "mismatches", "isup0_mismatches", "isup0_mismatches_v1_at_0p05"
    )}, indent=2))
    return out


def soft_target_for_slide(
    hard: np.ndarray,
    *,
    expected: set[int],
    metadata_isup: int,
    metadata_gleason: str,
    derived_isup: int,
    cancer_fracs: dict[int, float],
) -> tuple[np.ndarray, int, int, int, str]:
    """Build (H,W,C) soft targets. Returns soft, n_b, n_c, n_benign, rule."""
    h, w = hard.shape
    soft = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for c in range(NUM_CLASSES):
        soft[..., c] = (hard == c).astype(np.float32)

    protect = expected | {0, 1}
    cand = ~np.isin(hard, list(protect))
    n_cand = int(cand.sum())
    if n_cand == 0:
        return soft, 0, 0, 0, "none"

    n_benign = int(((hard == 2) & cand).sum())
    is_swap = (metadata_isup == 2 and derived_isup == 3) or (
        metadata_isup == 3 and derived_isup == 2
    )

    if is_swap:
        adjusted = {c: float(cancer_fracs.get(c, 0.0)) for c in expected}
        total = sum(adjusted.values())
        if total > 0:
            dist = np.zeros(NUM_CLASSES, dtype=np.float32)
            for c, v in adjusted.items():
                dist[int(c)] = v / total
            soft[cand] = dist
            return soft, n_cand, 0, n_benign, "B"
        # fallback hard primary
        primary = primary_gleason_class(metadata_gleason)
        dist = np.zeros(NUM_CLASSES, dtype=np.float32)
        dist[primary] = 1.0
        soft[cand] = dist
        return soft, 0, n_cand, n_benign, "C"

    primary = primary_gleason_class(metadata_gleason)
    dist = np.zeros(NUM_CLASSES, dtype=np.float32)
    dist[primary] = 1.0
    soft[cand] = dist
    return soft, 0, n_cand, n_benign, "C"


def apply_corrections(report: pd.DataFrame) -> dict:
    split = pd.read_csv(TRAIN_SPLIT)
    if "image_id" not in split.columns and "slide_id" in split.columns:
        split = split.rename(columns={"slide_id": "image_id"})

    mm = report[(~report.match) & (report.metadata_isup >= 1)].copy()
    slide_meta = {str(r.slide_id): r for _, r in mm.iterrows()}
    target_slides = set(slide_meta.keys())
    patches = split[split["image_id"].astype(str).isin(target_slides)].copy()
    print(f"ISUP1-5 mismatched slides: {len(target_slides)}")
    print(f"Patches to scan: {len(patches)}")

    if CORRECTED_DIR.exists():
        # clean prior partial runs for these slides only
        pass
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    stats = {
        "slides_rule_b": set(),
        "slides_rule_c": set(),
        "pixels_rule_b": 0,
        "pixels_rule_c": 0,
        "pixels_benign_corrected": 0,
        "patches_written": 0,
        "patches_scanned": 0,
        "by_isup": {
            i: {"slides_b": set(), "slides_c": set(), "pix_b": 0, "pix_c": 0}
            for i in range(1, 6)
        },
    }

    for slide_id, grp in tqdm(patches.groupby("image_id"), desc="correct"):
        sid = str(slide_id)
        meta = slide_meta[sid]
        expected = gleason_to_classes(str(meta.metadata_gleason))
        cancer_fracs = {
            3: float(meta.cancer_frac_g3),
            4: float(meta.cancer_frac_g4),
            5: float(meta.cancer_frac_g5),
        }
        h5_path = H5_DIR / f"{sid}_kept_raw.h5"
        if not h5_path.exists():
            print(f"WARNING: missing H5 {h5_path}")
            continue

        slide_dir = CORRECTED_DIR / sid
        with h5py.File(h5_path, "r") as f:
            masks = f["masks"]
            coords = np.asarray(f["coords"][:], dtype=np.int64)
            coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}

            for _, prow in grp.iterrows():
                x, y = int(prow["x"]), int(prow["y"])
                stats["patches_scanned"] += 1
                idx = coord_to_idx.get((x, y))
                if idx is None:
                    continue
                hard = np.clip(np.asarray(masks[idx]), 0, 5).astype(np.int64)
                soft, n_b, n_c, n_benign, rule = soft_target_for_slide(
                    hard,
                    expected=expected,
                    metadata_isup=int(meta.metadata_isup),
                    metadata_gleason=str(meta.metadata_gleason),
                    derived_isup=int(meta.derived_isup),
                    cancer_fracs=cancer_fracs,
                )
                n_corr = n_b + n_c
                if n_corr > 0:
                    slide_dir.mkdir(parents=True, exist_ok=True)
                    np.save(slide_dir / f"{x}_{y}.npy", soft)
                    stats["patches_written"] += 1
                    misup = int(meta.metadata_isup)
                    if n_b > 0:
                        stats["slides_rule_b"].add(sid)
                        stats["pixels_rule_b"] += n_b
                        stats["by_isup"][misup]["slides_b"].add(sid)
                        stats["by_isup"][misup]["pix_b"] += n_b
                    if n_c > 0:
                        stats["slides_rule_c"].add(sid)
                        stats["pixels_rule_c"] += n_c
                        stats["by_isup"][misup]["slides_c"].add(sid)
                        stats["by_isup"][misup]["pix_c"] += n_c
                    stats["pixels_benign_corrected"] += n_benign

                manifest_rows.append(
                    {
                        "slide_id": sid,
                        "x": x,
                        "y": y,
                        "num_pixels_corrected": int(n_corr),
                        "num_pixels_rule_b": int(n_b),
                        "num_pixels_rule_c": int(n_c),
                        "num_benign_pixels_corrected": int(n_benign),
                        "correction_rule_used": rule if n_corr > 0 else "none",
                        "metadata_isup": int(meta.metadata_isup),
                        "derived_isup": int(meta.derived_isup),
                        "metadata_gleason": meta.metadata_gleason,
                    }
                )

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)

    total_train_pixels = len(split) * 512 * 512
    total_corr_pix = stats["pixels_rule_b"] + stats["pixels_rule_c"]
    summary = {
        "min_area_pct": MIN_AREA_PCT,
        "checkpoint": str(
            PROJECT
            / "outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth"
        ),
        "mismatched_slides_isup1_5": int(len(target_slides)),
        "rule_b_slides": int(len(stats["slides_rule_b"])),
        "rule_b_pixels": int(stats["pixels_rule_b"]),
        "rule_c_slides": int(len(stats["slides_rule_c"])),
        "rule_c_pixels": int(stats["pixels_rule_c"]),
        "benign_class2_pixels_corrected": int(stats["pixels_benign_corrected"]),
        "patches_written": int(stats["patches_written"]),
        "patches_scanned": int(stats["patches_scanned"]),
        "total_pixels_corrected": int(total_corr_pix),
        "pct_of_all_train_pixels": float(100.0 * total_corr_pix / max(total_train_pixels, 1)),
        "isup0_total": int((report.metadata_isup == 0).sum()),
        "isup0_mismatches_v1": int(((report.metadata_isup == 0) & (~report.match)).sum()),
        "isup0_skipped": True,
        "by_isup": {
            str(i): {
                "slides_rule_b": len(stats["by_isup"][i]["slides_b"]),
                "slides_rule_c": len(stats["by_isup"][i]["slides_c"]),
                "pixels_rule_b": int(stats["by_isup"][i]["pix_b"]),
                "pixels_rule_c": int(stats["by_isup"][i]["pix_c"]),
            }
            for i in range(1, 6)
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("Correction summary (ISUP 1-5 mismatches only; ISUP-0 skipped entirely):")
    print(f"  Total mismatched slides (ISUP 1-5): {summary['mismatched_slides_isup1_5']}")
    print(
        f"  Rule B (validated 2<->3 swap):       {summary['rule_b_slides']} slides, "
        f"{summary['rule_b_pixels']} pixels corrected"
    )
    print(
        f"  Rule C (hard correction):            {summary['rule_c_slides']} slides, "
        f"{summary['rule_c_pixels']} pixels corrected"
    )
    print("\nBy ISUP grade (1-5 only):")
    print(f"  {'ISUP':>4} {'slides_B':>9} {'slides_C':>9} {'pix_B':>14} {'pix_C':>14}")
    for i in range(1, 6):
        b = summary["by_isup"][str(i)]
        print(
            f"  {i:4d} {b['slides_rule_b']:9d} {b['slides_rule_c']:9d} "
            f"{b['pixels_rule_b']:14d} {b['pixels_rule_c']:14d}"
        )
    print(
        f"\nISUP-0 slides: {summary['isup0_total']} total, "
        f"{summary.get('isup0_mismatches_v1', summary.get('isup0_mismatches_v2', '?'))} "
        f"mismatches at min_area_pct={MIN_AREA_PCT}; ALL skipped (no correction)."
    )
    print(
        f"\nTotal pixels corrected: {summary['total_pixels_corrected']} "
        f"({summary['pct_of_all_train_pixels']:.4f}% of all training pixels)"
    )
    print(
        f"Literal plan note: skip set is {{0,1}} only — benign(class2) pixels "
        f"corrected: {summary['benign_class2_pixels_corrected']}"
    )
    print("\n*** STOP — awaiting confirmation before retraining (Step 6). ***")
    print("=" * 72)


def load_diagnostic_v1() -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_csv(REPORT_V1)
    df["match"] = df["match"].astype(str).str.lower().isin(["true", "1"])
    return df


def main() -> None:
    report = load_diagnostic_v1()
    summary = apply_corrections(report)
    print_summary(summary)


if __name__ == "__main__":
    main()
