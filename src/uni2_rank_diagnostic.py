#!/usr/bin/env python3
"""Score UNI2 embeddings at full 4683 scale using rank, not absolute cosine.

The full run showed absolute cosine is scale-dependent and useless: median top-1
is 0.977, higher than the twin median measured in a 315-slide pool. Rank and
mutual-nearest-neighbour are relative, so they survive the pool growing.

The operational question is not AUC. It is: if we group by mutual-NN, do the
slides left with no mutual partner form a big enough, grade-balanced pool to be
val+test, and do confirmed twins reliably land inside a group rather than in
that pool?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
NEED = 940  # val + test


def main() -> None:
    d = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    sim = d["vecs"] @ d["vecs"].T
    np.fill_diagonal(sim, -1.0)
    n = len(ids)

    order = np.argsort(-sim, axis=1)
    rank_of = np.empty_like(order)
    rows = np.arange(n)[:, None]
    rank_of[rows, order] = np.arange(1, n + 1)[None, :]
    nn = order[:, 0]
    mutual = nn[nn] == rows[:, 0]

    pos, neg = load_labels()

    def stats(pairs, name):
        rows_out = []
        for a, b in pairs:
            if a not in idx or b not in idx:
                continue
            i, j = idx[a], idx[b]
            rows_out.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "cos": float(sim[i, j]),
                    "rank": int(min(rank_of[i, j], rank_of[j, i])),
                    "mutual_nn": bool(nn[i] == j and nn[j] == i),
                }
            )
        df = pd.DataFrame(rows_out)
        df.to_csv(G / f"uni2_fullscale_{name}.csv", index=False)
        return df

    p, q = stats(pos, "twins"), stats(neg, "not_twins")
    print(f"\nbenchmark at full scale: {len(p)} confirmed twins | {len(q)} confirmed not-twins")
    print(f"{'':16} {'cos med':>9} {'rank med':>9} {'rank<=1':>9} {'rank<=5':>9} {'mutual':>9}")
    for name, df in (("TWINS", p), ("NOT-TWINS", q)):
        if df.empty:
            continue
        print(
            f"{name:16} {df.cos.median():9.4f} {df['rank'].median():9.1f} "
            f"{(df['rank'] <= 1).mean():9.1%} {(df['rank'] <= 5).mean():9.1%} {df.mutual_nn.mean():9.1%}"
        )

    # What does the mutual-NN rule actually flag, and what is left over?
    mpairs = {tuple(sorted([ids[i], ids[nn[i]]])) for i in range(n) if mutual[i]}
    posset, negset = set(pos), set(neg)
    print(f"\nmutual-NN pairs: {len(mpairs)}")
    print(f"  confirmed twin     {len(mpairs & posset):5d}")
    print(f"  confirmed not-twin {len(mpairs & negset):5d}")
    print(f"  unknown            {len(mpairs - posset - negset):5d}")

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    grouped = {s for pr in mpairs for s in pr}
    free = [i for i in ids if i not in grouped]
    print(f"\nslides with no mutual partner (val/test candidate pool): {len(free)} / {n}")
    g = meta.loc[free].isup_grade.value_counts().sort_index()
    share = meta.loc[ids].isup_grade.value_counts().sort_index()
    print(f"{'ISUP':>5} {'free':>6} {'total':>6} {'free %':>8} {'need':>6}")
    ok = True
    for k in share.index:
        need = int(round(NEED * share[k] / n))
        if g.get(k, 0) < need:
            ok = False
        print(f"{k:5d} {g.get(k,0):6d} {share[k]:6d} {g.get(k,0)/share[k]:8.1%} {need:6d}")

    # A twin landing in the free pool is the only expensive error.
    leaked = p[~p.id_a.isin(grouped) | ~p.id_b.isin(grouped)] if not p.empty else p
    report = {
        "n_mutual_pairs": len(mpairs),
        "twin_mutual_recall": float(p.mutual_nn.mean()) if not p.empty else None,
        "twin_rank1_recall": float((p["rank"] <= 1).mean()) if not p.empty else None,
        "not_twin_mutual_rate": float(q.mutual_nn.mean()) if not q.empty else None,
        "free_pool": len(free),
        "free_pool_enough_per_grade": ok,
        "confirmed_twins_with_a_slide_in_free_pool": int(len(leaked)),
    }
    (G / "uni2_rank_diagnostic.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
