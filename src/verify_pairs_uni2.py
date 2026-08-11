#!/usr/bin/env python3
"""Ask UNI2 embeddings about a specific list of pairs.

Distractors are embedded alongside the pairs so a cosine can be read as a rank,
not just a number: "0.97" only means something if we know how many unrelated
slides also score that high.

Calibration comes from the 315-slide smoke test (job 5393822):
  confirmed twins      median 0.976, min 0.828
  confirmed not-twins  median 0.855, max 0.942
so 0.942 is the highest score any pair the user cleared ever reached.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from smoke_uni2_dedupe import embed, load_patch_index

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
NOT_TWIN_CEILING = 0.942
TWIN_MEDIAN = 0.976


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=str(G / "page153_pairs.csv"))
    ap.add_argument("--n-random", type=int, default=250)
    ap.add_argument("--n-patch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="page153")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, dtype=str)
    involved = sorted(set(pairs.keep_id) | set(pairs.drop_id))

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str})
    rng = np.random.default_rng(args.seed)
    pool = [i for i in meta.image_id if i not in set(involved)]
    distract = list(rng.choice(pool, size=min(args.n_random, len(pool)), replace=False))
    ids = sorted(set(involved) | set(distract))
    print(f"{len(pairs)} pairs | {len(involved)} slides of interest | {len(distract)} distractors", flush=True)

    ids, vecs = embed(ids, load_patch_index(), args.n_patch, "cuda", args.seed)
    idx = {s: i for i, s in enumerate(ids)}
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    np.savez_compressed(G / f"verify_{args.tag}.npz", ids=np.array(ids), sim=sim)

    rows = []
    for r in pairs.itertuples(index=False):
        if r.keep_id not in idx or r.drop_id not in idx:
            continue
        i, j = idx[r.keep_id], idx[r.drop_id]
        cos = float(sim[i, j])
        rank = int((sim[i] > cos).sum()) + 1
        top = ids[int(sim[i].argmax())]
        verdict = "TWIN" if cos > NOT_TWIN_CEILING else ("borderline" if cos > 0.90 else "not twin")
        rows.append(
            {
                "pair": r.pair_num,
                "row": r.row,
                "a": r.keep_id[:8],
                "b": r.drop_id[:8],
                "shape_iou": r.shape_iou,
                "uni2_cos": round(cos, 4),
                "rank_of_b_for_a": rank,
                "a_top1": top[:8],
                "a_top1_is_b": top == r.drop_id,
                "verdict": verdict,
            }
        )
    out = pd.DataFrame(rows).sort_values("uni2_cos", ascending=False)
    out.to_csv(G / f"verify_{args.tag}_results.csv", index=False)
    print(f"\ncalibration: not-twin ceiling {NOT_TWIN_CEILING}, twin median {TWIN_MEDIAN}, "
          f"{len(ids)} slides embedded\n")
    print(out.to_string(index=False))
    print(f"\nTWIN {int((out.verdict=='TWIN').sum())} | borderline "
          f"{int((out.verdict=='borderline').sum())} | not twin {int((out.verdict=='not twin').sum())}")


if __name__ == "__main__":
    main()
