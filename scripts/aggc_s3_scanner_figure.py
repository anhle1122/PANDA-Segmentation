#!/usr/bin/env python3
"""Subset3 paired figure: specimen on x, one point per scanner."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

ID_RE = re.compile(r"Subset3_(?:Train|Test)_(\d+)_(.+)$")
SCANNER_ORDER = ["Akoya", "KFBio", "Leica", "Olympus", "Philips", "Zeiss"]


def parse_id(sid: str) -> tuple[str, str] | None:
    m = ID_RE.match(sid)
    if not m:
        return None
    return m.group(1), m.group(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--metric", default="weighted_f1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Subset3 Test native raw — per specimen, one point per scanner")
    args = p.parse_args()

    rows = list(csv.DictReader(args.csv.open()))
    by_case: dict[str, dict[str, float]] = {}
    for r in rows:
        parsed = parse_id(r["slide_id"])
        if parsed is None:
            continue
        case, scanner = parsed
        try:
            val = float(r[args.metric])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        by_case.setdefault(case, {})[scanner] = val
    cases = sorted(by_case, key=lambda c: int(c))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cases) + 2), 4.8))
    xs = np.arange(len(cases))
    markers = ["o", "s", "D", "^", "v", "P"]
    for i, sc in enumerate(SCANNER_ORDER):
        ys, xsc = [], []
        for x, case in zip(xs, cases):
            if sc in by_case[case]:
                xsc.append(x)
                ys.append(by_case[case][sc])
        if ys:
            ax.scatter(xsc, ys, marker=markers[i % len(markers)], label=sc, s=36, zorder=3)
        # connect scanners of the same specimen
    for x, case in zip(xs, cases):
        pts = [(x, by_case[case][sc]) for sc in SCANNER_ORDER if sc in by_case[case]]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="0.75", lw=0.8, zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cases, rotation=0)
    ax.set_xlabel("Specimen id")
    ax.set_ylabel(args.metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.set_title(args.title)
    ax.legend(ncols=3, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out} n_specimens={len(cases)}", flush=True)


if __name__ == "__main__":
    main()
