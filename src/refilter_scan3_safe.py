#!/usr/bin/env python3
"""Drop newly-ledgered not-twin edges from the scan3 safe bucket and re-render it.

Safe pairs are 2-member components, so cutting one of their edges removes the
pair outright — no other cluster can change shape. That lets us patch the folder
without paying for a full rescan + lower-IoU re-render.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from patch_utils import PROJECT
from render_dedupe_galleries import render_pair_pages
from twin_ledger import adjudicated_pairs

OUT = PROJECT / "outputs" / "docs" / "slide_duplicates_scan3_clean"


def main() -> None:
    judged = adjudicated_pairs()
    safe = pd.read_csv(OUT / "dedupe_safe_pairs_iou70.csv", dtype={"image_id_a": str, "image_id_b": str})
    edge = safe.apply(lambda r: tuple(sorted([r.image_id_a, r.image_id_b])), axis=1)
    hit = edge.isin(judged)
    if not hit.any():
        print("nothing to remove")
        return

    removed = safe[hit]
    removed.to_csv(OUT / "safe_pairs_removed_already_judged.csv", index=False)
    safe = safe[~hit].reset_index(drop=True)
    safe["cluster_id"] = range(1, len(safe) + 1)
    safe.to_csv(OUT / "dedupe_safe_pairs_iou70.csv", index=False)

    gal = OUT / "galleries" / "safe_pairs"
    for old in gal.glob("*.png"):
        old.unlink()
    render_pair_pages(safe, gal, "SAFE IoU>=0.70")

    summary = json.loads((OUT / "rescan_summary.json").read_text())
    summary["n_safe_iou70"] = len(safe)
    summary["n_suppressed_already_judged_edges"] += int(hit.sum())
    (OUT / "rescan_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"removed {int(hit.sum())} judged safe pairs -> {len(safe)} left")


if __name__ == "__main__":
    main()
