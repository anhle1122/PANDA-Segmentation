#!/usr/bin/env python3
"""Embed every slide with UNI2, then report how isolated each one is.

Isolation, not grouping, is what the split needs: we pick val/test from the
slides whose nearest neighbour is furthest away, so no close call ever has to be
adjudicated. Grouping errors are free here -- a wrongly grouped slide just goes
to train.

Ranking is done per ISUP grade because benign cores resemble each other far more
than tumour cores do; a global ranking would leave val/test with almost no
ISUP 0.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from smoke_uni2_dedupe import embed, load_patch_index

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
NEED_PER_SPLIT = 470  # ~10% of 4683, for val and for test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    G.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str})
    ids_in = meta.image_id.tolist()
    print(f"embedding {len(ids_in)} slides at {args.n_patch} patches each", flush=True)

    cache = G / "uni2_slide_embeddings.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        ids, vecs = [str(x) for x in d["ids"]], d["vecs"]
        print(f"loaded cached embeddings for {len(ids)} slides", flush=True)
    else:
        ids, vecs = embed(ids_in, load_patch_index(), args.n_patch, "cuda", args.seed)
        np.savez_compressed(cache, ids=np.array(ids), vecs=vecs)
        print(f"wrote {cache}", flush=True)

    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)

    part = np.partition(sim, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    nn = sim.argmax(1)
    isup = meta.set_index("image_id").isup_grade.astype(int)

    df = pd.DataFrame(
        {
            "image_id": ids,
            "isup_grade": [isup[i] for i in ids],
            "top1_cos": top1.round(4),
            "top2_cos": top2.round(4),
            "margin": (top1 - top2).round(4),
            "nn_id": [ids[j] for j in nn],
            "mutual": [bool(nn[nn[i]] == i) for i in range(len(ids))],
        }
    )
    df["isolation_rank_in_grade"] = df.groupby("isup_grade").top1_cos.rank(method="first")
    df.sort_values("top1_cos").to_csv(G / "uni2_slide_isolation.csv", index=False)

    print("\ntop1 cosine distribution (lower = more isolated = safer for val/test):")
    print(df.top1_cos.describe().round(4).to_string())

    print("\nper ISUP grade, cosine of the Nth most isolated slide:")
    print(f"{'ISUP':>5} {'n':>6} " + " ".join(f"{f'@{n}':>8}" for n in (100, 200, 300, 470)))
    feasible = True
    for g, sub in df.groupby("isup_grade"):
        s = sub.top1_cos.sort_values().to_numpy()
        cells = []
        for n in (100, 200, 300, 470):
            cells.append(f"{s[n-1]:8.4f}" if len(s) >= n else f"{'--':>8}")
        need = int(round(NEED_PER_SPLIT * 2 * len(sub) / len(df)))
        if len(s) < need:
            feasible = False
        print(f"{g:5d} {len(sub):6d} " + " ".join(cells) + f"   need {need} for val+test")

    report = {
        "n_slides": len(ids),
        "median_top1": float(np.median(top1)),
        "n_mutual_pairs": int(df.mutual.sum() // 2),
        "stratified_feasible": feasible,
    }
    (G / "uni2_isolation_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
