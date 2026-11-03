#!/usr/bin/env python3
"""Per-slide median / IQR and paired bootstrap for AGGC slide CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "weighted_f1",
    "dice_Stroma",
    "dice_Normal",
    "dice_G3",
    "dice_G4",
    "dice_G5",
]


def median_iqr(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
    }


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, seed: int = 0
) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {"n": 0}
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = d.size
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "mean_diff_b_minus_a": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_boot_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
    }


def summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {"path": str(path), "n_slides": int(len(df)), "metrics": {}}
    for m in METRICS:
        if m not in df.columns:
            continue
        out["metrics"][m] = median_iqr(df[m].to_numpy())
    return out, df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_a", type=Path, help="Per-slide CSV (model A / raw)")
    p.add_argument("csv_b", type=Path, nargs="?", help="Optional paired model B CSV")
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    report_a, df_a = summarize_csv(args.csv_a)
    report = {"A": report_a}
    if args.csv_b:
        report_b, df_b = summarize_csv(args.csv_b)
        report["B"] = report_b
        merged = df_a.merge(df_b, on="slide_id", suffixes=("_a", "_b"))
        report["paired_n"] = int(len(merged))
        report["paired_bootstrap"] = {}
        for m in METRICS:
            ca, cb = f"{m}_a", f"{m}_b"
            if ca not in merged.columns or cb not in merged.columns:
                continue
            report["paired_bootstrap"][m] = paired_bootstrap(
                merged[ca].to_numpy(), merged[cb].to_numpy(), n_boot=args.n_boot
            )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
