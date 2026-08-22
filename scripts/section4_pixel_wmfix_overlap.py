#!/usr/bin/env python3
"""Section 4: pixel-level overlap of ep14 referee swaps vs original wmfix flags.

wmfix flagged pixel = teacher-A argmax in that slide's flag_pred_classes
(same definition as Round 1 train). Referee swapped pixel = corrected
target != original mask on a non-ignored write.

Does not train.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
WMFIX = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"
WMFIX_PRED = PROJECT / "outputs" / "pseudo_label" / "round1_source_pred"
REF_DIR = PROJECT / "outputs" / "pseudo_label" / "corrections_opt3_omar6_locked_locked_r2_ep014"
CORRECTING = {"rule1_soft_tie", "rule2_adjacent_invented", "rule3_invented_default"}


def parse_flags(raw) -> set[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    return {int(x) for x in text.split("|") if x != ""}


def coord_index(coords: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--referee-dir", type=Path, default=REF_DIR)
    ap.add_argument("--wmfix", type=Path, default=WMFIX)
    ap.add_argument("--wmfix-pred", type=Path, default=WMFIX_PRED)
    ap.add_argument("--split", type=Path, default=PROJECT / "outputs" / "splits" / "panda_train.csv")
    ap.add_argument("--max-slides", type=int, default=None)
    args = ap.parse_args()

    from train.baseline_dataset import BaselinePatchDataset

    wm = pd.read_csv(args.wmfix, dtype={"slide_id": str})
    ref = pd.read_csv(args.referee_dir / "correction_manifest.csv", dtype={"slide_id": str})
    wm["correcting"] = wm["rule_applied"].isin(CORRECTING)
    ref["n_swap"] = ref.get("n_swap", 0).fillna(0)
    correcting = wm[wm["correcting"]].copy()
    if args.max_slides:
        correcting = correcting.head(args.max_slides)

    ds = BaselinePatchDataset(args.split, mode="raw", allow_missing_h5=True)

    n_wmfix_px = 0
    n_ref_px_on_shared = 0
    n_overlap_px = 0
    n_wmfix_only_px = 0
    n_ref_only_shared_px = 0
    n_shared_patches = 0
    n_slides_both_files = 0
    n_wmfix_missing_pred = 0
    n_ref_missing_h5 = 0
    by_rule = {r: {"slides": 0, "wmfix_px": 0, "overlap_px": 0} for r in CORRECTING}
    rows = []

    for i, row in enumerate(correcting.itertuples(), start=1):
        sid = str(row.slide_id)
        flags = parse_flags(getattr(row, "flag_pred_classes", ""))
        rule = str(row.rule_applied)
        a_path = args.wmfix_pred / f"{sid}_srcpred.h5"
        r_path = args.referee_dir / f"{sid}_corrected.h5"
        rec = {
            "slide_id": sid,
            "rule_applied": rule,
            "wmfix_px": 0,
            "ref_swap_px_shared": 0,
            "overlap_px": 0,
            "shared_patches": 0,
            "skipped": "",
        }
        if not flags:
            rec["skipped"] = "no_flag_classes"
            rows.append(rec)
            continue
        if not a_path.is_file():
            n_wmfix_missing_pred += 1
            rec["skipped"] = "missing_teacherA_h5"
            rows.append(rec)
            continue
        if not r_path.is_file():
            n_ref_missing_h5 += 1
            rec["skipped"] = "no_referee_corrected_h5"
            # still count wmfix pixels if we can
            with h5py.File(a_path, "r") as fa:
                pred_a = np.asarray(fa["preds"])
            rec["wmfix_px"] = int(np.isin(pred_a, list(flags)).sum())
            n_wmfix_px += rec["wmfix_px"]
            n_wmfix_only_px += rec["wmfix_px"]
            by_rule[rule]["slides"] += 1
            by_rule[rule]["wmfix_px"] += rec["wmfix_px"]
            rows.append(rec)
            continue

        n_slides_both_files += 1
        with h5py.File(a_path, "r") as fa, h5py.File(r_path, "r") as fr:
            coords_a = np.asarray(fa["coords"])
            pred_a = np.asarray(fa["preds"])
            coords_r = np.asarray(fr["coords"])
            target = np.asarray(fr["target"])
        idx_a = coord_index(coords_a)
        wmfix_slide = 0
        ref_slide = 0
        overlap_slide = 0
        shared = 0
        for j, (x, y) in enumerate(coords_r):
            key = (int(x), int(y))
            ia = idx_a.get(key)
            if ia is None:
                continue
            shared += 1
            mask = ds._read_mask(sid, key[0], key[1])
            wm_hit = np.isin(pred_a[ia], list(flags))
            ref_hit = target[j] != mask
            wmfix_slide += int(wm_hit.sum())
            ref_slide += int(ref_hit.sum())
            overlap_slide += int((wm_hit & ref_hit).sum())
        n_shared_patches += shared
        n_wmfix_px += wmfix_slide
        n_ref_px_on_shared += ref_slide
        n_overlap_px += overlap_slide
        n_wmfix_only_px += wmfix_slide - overlap_slide
        n_ref_only_shared_px += ref_slide - overlap_slide
        by_rule[rule]["slides"] += 1
        by_rule[rule]["wmfix_px"] += wmfix_slide
        by_rule[rule]["overlap_px"] += overlap_slide
        rec.update(
            {
                "wmfix_px": wmfix_slide,
                "ref_swap_px_shared": ref_slide,
                "overlap_px": overlap_slide,
                "shared_patches": shared,
            }
        )
        rows.append(rec)
        if i % 25 == 0:
            print(
                f"  {i}/{len(correcting)} slides  overlap_px={n_overlap_px} wmfix_px={n_wmfix_px}",
                flush=True,
            )

    ref_swap_all = int(ref["n_swap"].sum())
    payload = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": {
            "wmfix_pixel": "teacher-A pred in flag_pred_classes (Round 1 train flag_mask)",
            "referee_pixel": "corrected target != original mask on shared (x,y) patches",
            "correcting_rules": sorted(CORRECTING),
        },
        "n_wmfix_correcting_slides": int(len(wm[wm["correcting"]])),
        "n_slides_scanned": int(len(correcting)),
        "n_slides_both_files": n_slides_both_files,
        "n_wmfix_missing_pred": n_wmfix_missing_pred,
        "n_ref_missing_h5": n_ref_missing_h5,
        "n_shared_patches": n_shared_patches,
        "n_wmfix_flagged_pixels": n_wmfix_px,
        "n_referee_swap_pixels_all_slides": ref_swap_all,
        "n_referee_swap_pixels_on_shared_patches": n_ref_px_on_shared,
        "n_overlap_pixels": n_overlap_px,
        "n_wmfix_only_pixels": n_wmfix_only_px,
        "n_referee_only_on_shared_patches": n_ref_only_shared_px,
        "frac_wmfix_also_referee": (n_overlap_px / n_wmfix_px) if n_wmfix_px else None,
        "frac_shared_ref_also_wmfix": (n_overlap_px / n_ref_px_on_shared) if n_ref_px_on_shared else None,
        "frac_all_ref_also_wmfix": (n_overlap_px / ref_swap_all) if ref_swap_all else None,
        "by_rule": by_rule,
        "note": (
            "Low pixel overlap is expected: wmfix rewrites teacher-A flagged "
            "classes (including legal-grade Rule 1 soft ties); referee only "
            "swaps high-conf illegal G3/G4/G5."
        ),
    }
    out_dir = args.referee_dir
    (out_dir / "section4_pixel_wmfix_overlap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(out_dir / "section4_pixel_wmfix_overlap.csv", index=False)
    lines = [
        "# Section 4 — pixel-level wmfix vs ep14 referee",
        "",
        f"Scanned {payload['n_slides_scanned']} wmfix-correcting slides "
        f"({n_slides_both_files} had both H5s).",
        "",
        "| | pixels |",
        "|---|---:|",
        f"| wmfix flagged | {n_wmfix_px:,} |",
        f"| referee swap (all slides) | {ref_swap_all:,} |",
        f"| referee swap on shared patches | {n_ref_px_on_shared:,} |",
        f"| **overlap** | **{n_overlap_px:,}** |",
        f"| wmfix only | {n_wmfix_only_px:,} |",
        f"| referee only (shared patches) | {n_ref_only_shared_px:,} |",
        "",
        f"Of wmfix flagged pixels, **{(payload['frac_wmfix_also_referee'] or 0):.1%}** "
        "are also referee swaps.",
        f"Of all referee swaps, **{(payload['frac_all_ref_also_wmfix'] or 0):.1%}** "
        "overlap a wmfix flag.",
        "",
        payload["note"],
        "",
        "No training started.",
        "",
    ]
    (out_dir / "section4_pixel_wmfix_overlap.md").write_text("\n".join(lines), encoding="utf-8")
    docs = PROJECT / "outputs" / "docs" / "opt3_this_run" / "section4_pixel_wmfix_overlap.md"
    docs.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "by_rule"}, indent=2))
    print("Wrote", out_dir / "section4_pixel_wmfix_overlap.md")


if __name__ == "__main__":
    main()
