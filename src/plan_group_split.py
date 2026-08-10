#!/usr/bin/env python3
"""Turn a chosen duplicate signal into the operational number that matters.

Benchmark FPR is measured against *hard* negatives -- pairs similar enough that
they were put in front of the user -- so it badly overstates the real cost. What
decides the split is simpler: at threshold t, how many slides end up attached to
some other slide (forced to train) and how many stay isolated and can be spent
on val/test?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
NEEDED = 940  # ~10% val + ~10% test of 4683


def components(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def main() -> None:
    sim = np.load(G / "content_similarity.npz", allow_pickle=True)
    ids = [str(x) for x in sim["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    hist = sim["hist"]
    n = len(ids)

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str})
    isup = meta.set_index("image_id").isup_grade.astype(int)
    overall = isup.value_counts(normalize=True).sort_index()

    pos, _ = load_labels()
    pos = [(x, y) for x, y in pos if x in idx and y in idx]
    pv = np.array([hist[idx[x], idx[y]] for x, y in pos])

    print(f"{'recall':>7} {'thresh':>7} {'edges':>9} {'in groups':>10} {'isolated':>9} "
          f"{'biggest':>8} {'val/test ok?':>13} {'max ISUP dev':>13}")
    for rec in (0.999, 0.99, 0.98, 0.95, 0.90, 0.80):
        t = float(np.percentile(pv, 100 * (1 - rec)))
        a, b = np.where(np.triu(hist >= t, 1))
        roots = components(n, np.column_stack([a, b]))
        sizes = pd.Series(roots).value_counts()
        singleton_roots = set(sizes[sizes == 1].index)
        iso = [ids[i] for i in range(n) if roots[i] in singleton_roots]
        s = pd.Series([isup[i] for i in iso]).value_counts(normalize=True).sort_index()
        dev = float((s - overall).abs().max() * 100) if len(iso) else 99.9
        print(f"{rec:7.1%} {t:7.3f} {len(a):9d} {n-len(iso):10d} {len(iso):9d} "
              f"{sizes.max():8d} {'YES' if len(iso)>=NEEDED else 'NO':>13} {dev:12.2f}pp")


if __name__ == "__main__":
    main()
