#!/usr/bin/env python3
"""Section 2: maxprob histograms on the locked ep14 teacher pack.

Two populations (non–ISUP-0):
  A) all pred != mask
  B) pred in {3,4,5} and pred not in clinical {primary, secondary}

Does not train. Does not write a correction dir.
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
TEACHER = PROJECT / "outputs" / "pseudo_label" / "teacher_opt3_omar6_locked_locked_r2_ep014"
OUT = PROJECT / "outputs" / "pseudo_label" / "corrections_opt3_omar6_locked_locked_r2_ep014"
BINS = np.linspace(0.0, 1.0, 21)
TAU = 0.7


def summarize(name: str, n: int, n_ge_tau: int, hist: np.ndarray, acc: dict) -> dict:
    if n == 0:
        return {"name": name, "n": 0}
    # Approximate median from 10k-bin CDF stored as running hist
    cdf = np.cumsum(hist)
    mid = n / 2.0
    idx = int(np.searchsorted(cdf, mid, side="left"))
    idx = min(idx, len(BINS) - 2)
    # linear interpolate inside bin
    prev = float(cdf[idx - 1]) if idx else 0.0
    width = float(hist[idx]) or 1.0
    frac = (mid - prev) / width
    median = float(BINS[idx] + frac * (BINS[idx + 1] - BINS[idx]))
    return {
        "name": name,
        "n": int(n),
        "n_ge_tau": int(n_ge_tau),
        "frac_ge_tau": n_ge_tau / n,
        "median_approx_from_bins": median,
        "mean_approx": acc["sum"] / n,
        "hist_20bins_0_to_1": hist.astype(int).tolist(),
        "bin_edges": BINS.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", type=Path, default=TEACHER)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--split", type=Path, default=PROJECT / "outputs" / "splits" / "panda_train.csv")
    ap.add_argument("--clinical-csv", type=Path, default=None)
    ap.add_argument("--max-slides", type=int, default=None)
    args = ap.parse_args()

    from apply_isup_referee import load_clinical, DEFAULT_CLINICAL
    from train.baseline_dataset import BaselinePatchDataset
    from train.pseudo_label_rules import gleason_to_classes

    clinical_path = args.clinical_csv or (args.teacher_dir / "clinical_isup.csv")
    if not clinical_path.is_file():
        clinical_path = DEFAULT_CLINICAL
    clin = load_clinical(clinical_path).set_index("image_id")
    split = pd.read_csv(args.split, dtype={"image_id": str})
    slide_ids = sorted(split["image_id"].astype(str).unique())
    if args.max_slides:
        slide_ids = slide_ids[: args.max_slides]
    ds = BaselinePatchDataset(args.split, mode="raw", allow_missing_h5=True)

    hist_a = np.zeros(len(BINS) - 1, dtype=np.int64)
    hist_b = np.zeros(len(BINS) - 1, dtype=np.int64)
    n_a = n_b = 0
    n_a_hi = n_b_hi = 0
    acc_a = {"sum": 0.0}
    acc_b = {"sum": 0.0}
    n_slides = 0

    for i, sid in enumerate(slide_ids, start=1):
        if sid not in clin.index:
            continue
        if int(clin.loc[sid, "isup_grade"]) == 0:
            continue
        h5_path = args.teacher_dir / f"{sid}_srcpred.h5"
        if not h5_path.is_file():
            continue
        allowed = gleason_to_classes(str(clin.loc[sid, "gleason_score"]))
        with h5py.File(h5_path, "r") as f:
            coords = np.asarray(f["coords"])
            pred = np.asarray(f["preds"])
            maxprob = np.asarray(f["maxprob"], dtype=np.float32)
        masks = [ds._read_mask(sid, int(x), int(y)) for x, y in coords]
        mask = np.stack(masks, axis=0)
        disagree = pred != mask
        if disagree.any():
            mp = maxprob[disagree]
            hist_a += np.histogram(mp, bins=BINS)[0]
            n_a += int(mp.size)
            n_a_hi += int((mp >= TAU).sum())
            acc_a["sum"] += float(mp.sum())
        illegal = disagree & np.isin(pred, (3, 4, 5))
        if allowed:
            for c in (3, 4, 5):
                if c in allowed:
                    illegal &= pred != c
        else:
            illegal &= False
        if illegal.any():
            mp = maxprob[illegal]
            hist_b += np.histogram(mp, bins=BINS)[0]
            n_b += int(mp.size)
            n_b_hi += int((mp >= TAU).sum())
            acc_b["sum"] += float(mp.sum())
        n_slides += 1
        if i % 50 == 0:
            print(f"  {i}/{len(slide_ids)}  A={n_a} B={n_b}", flush=True)

    payload = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "teacher_dir": str(args.teacher_dir),
        "tau": TAU,
        "n_slides_scanned": n_slides,
        "all_pred_ne_mask": summarize("all_pred_ne_mask", n_a, n_a_hi, hist_a, acc_a),
        "isup_mismatch_cancer": summarize("isup_mismatch_cancer", n_b, n_b_hi, hist_b, acc_b),
        "prior_expectation": {
            "all_pred_ne_mask": "historically ~80% >=0.7, median ~0.98",
            "isup_mismatch_cancer": "historically ~51% >=0.7, median ~0.68",
        },
        "auto_train": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "section2_confidence_histograms.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    a = payload["all_pred_ne_mask"]
    b = payload["isup_mismatch_cancer"]
    lines = [
        "# Section 2 — ep14 maxprob histograms",
        "",
        f"Scanned {n_slides} non–ISUP-0 pack slides. τ = {TAU}. No training.",
        "",
        "| Population | pixels | share ≥ 0.7 | median (20-bin approx) |",
        "|---|---:|---:|---:|",
        f"| all pred≠mask | {a.get('n', 0):,} | {a.get('frac_ge_tau', 0):.1%} | {a.get('median_approx_from_bins', 0):.3f} |",
        f"| ISUP-mismatch cancer | {b.get('n', 0):,} | {b.get('frac_ge_tau', 0):.1%} | {b.get('median_approx_from_bins', 0):.3f} |",
        "",
        "Prior (other teachers): all-disagree ~80% / median ~0.98; ISUP-mismatch ~51% / median ~0.68.",
        "",
        "No training started.",
        "",
    ]
    text = "\n".join(lines)
    (args.out_dir / "section2_confidence_histograms.md").write_text(text, encoding="utf-8")
    docs = PROJECT / "outputs" / "docs" / "opt3_this_run" / "section2_confidence_histograms.md"
    docs.write_text(text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Wrote", args.out_dir / "section2_confidence_histograms.md")


if __name__ == "__main__":
    main()
