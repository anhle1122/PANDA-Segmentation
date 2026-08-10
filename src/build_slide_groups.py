#!/usr/bin/env python3
"""Grade-agnostic slide similarity graph → leakage groups.

The dedupe detector only ever compared slides inside the same
``(isup_grade, gleason_score)`` bucket, then kept mutual top-3 neighbours. That
makes two whole classes of duplicate invisible: serial sections graded
differently (the ISUP 2<->3 boundary), and extra members of a large cluster
whose edges lost the top-3 cut. For split safety we need every edge, so this
does the full 4683 x 4683 comparison with no label bucketing.

Brute force is fine if we stop looping in Python: flatten each silhouette, then
one matmul per orientation gives all intersection counts at once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dedupe_slides_shape_isup import FP_PATH, META_PATH
from patch_utils import PROJECT

OUT = PROJECT / "outputs" / "docs" / "slide_groups"

ORIENTS = (
    ("id", lambda b: b),
    ("fliplr", np.fliplr),
    ("flipud", np.flipud),
    ("rot180_flip", lambda b: np.fliplr(np.flipud(b))),
    ("rot90", lambda b: np.rot90(b, 1)),
    ("rot180", lambda b: np.rot90(b, 2)),
    ("rot270", lambda b: np.rot90(b, 3)),
    ("rot90_flip", lambda b: np.fliplr(np.rot90(b, 1))),
    ("rot270_flip", lambda b: np.fliplr(np.rot90(b, 3))),
)


def pairwise_best_iou(sils: np.ndarray) -> np.ndarray:
    """Max IoU over the 9 flip/rotation orientations, for every pair."""
    n = len(sils)
    base = (sils > 0).astype(np.float32)
    flat = base.reshape(n, -1)
    area = flat.sum(1)

    best = np.zeros((n, n), dtype=np.float32)
    for name, fn in ORIENTS:
        oriented = np.stack([fn(s) for s in base]).reshape(n, -1)
        inter = flat @ oriented.T
        union = area[:, None] + area[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        np.maximum(best, iou, out=best)
        print(f"  orientation {name} done", flush=True)
    np.fill_diagonal(best, 0.0)
    # IoU(a,b) must not depend on which side got rotated
    return np.maximum(best, best.T)


def components(ids: list[str], iou: np.ndarray, thr: float) -> dict[str, int]:
    """Union-find over edges >= thr."""
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(*np.where(np.triu(iou >= thr, 1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return {ids[i]: find(i) for i in range(len(ids))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-min", type=float, default=0.30, help="Edges below this are never written")
    ap.add_argument("--group-thr", type=float, default=0.50, help="Threshold used for the shipped group_id")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(FP_PATH)
    sils = data["sils"]
    meta = pd.read_csv(META_PATH)
    meta["image_id"] = meta["image_id"].astype(str)
    ids = meta.image_id.tolist()
    print(f"slides={len(ids)} silhouette={sils.shape[1:]}", flush=True)

    iou = pairwise_best_iou(sils)

    a, b = np.where(np.triu(iou >= args.edge_min, 1))
    edges = pd.DataFrame(
        {
            "image_id_a": [ids[i] for i in a],
            "image_id_b": [ids[j] for j in b],
            "shape_iou": iou[a, b].round(4),
            "isup_a": meta.isup_grade.to_numpy()[a],
            "isup_b": meta.isup_grade.to_numpy()[b],
        }
    ).sort_values("shape_iou", ascending=False)
    edges["cross_grade"] = edges.isup_a != edges.isup_b
    edges.to_csv(OUT / "all_pairs_grade_agnostic.csv", index=False)
    print(f"edges>={args.edge_min}: {len(edges)} ({int(edges.cross_grade.sum())} cross-grade)", flush=True)

    report = {"n_slides": len(ids), "edge_min": args.edge_min, "group_thr": args.group_thr, "thresholds": {}}
    for thr in (0.90, 0.70, 0.60, 0.50, 0.40, 0.30):
        gmap = components(ids, iou, thr)
        sizes = pd.Series(gmap).value_counts()
        n_single = int((sizes == 1).sum())
        sub = edges[edges.shape_iou >= thr]
        report["thresholds"][str(thr)] = {
            "n_edges": int(len(sub)),
            "n_cross_grade_edges": int(sub.cross_grade.sum()),
            "n_groups": int(len(sizes)),
            "n_singletons": n_single,
            "n_in_groups": len(ids) - n_single,
            "largest_group": int(sizes.max()),
        }
        print(f"  thr {thr}: groups {len(sizes)} singletons {n_single} largest {sizes.max()}", flush=True)

    gmap = components(ids, iou, args.group_thr)
    codes = {root: i for i, root in enumerate(sorted(set(gmap.values())))}
    out = meta[["image_id", "isup_grade", "gleason_score"]].copy()
    out["group_id"] = [codes[gmap[i]] for i in ids]
    size = out.group_id.value_counts()
    out["group_size"] = out.group_id.map(size)
    out["is_singleton"] = out.group_size == 1
    out.to_csv(OUT / "slide_groups.csv", index=False)

    (OUT / "group_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
