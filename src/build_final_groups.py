#!/usr/bin/env python3
"""Build slide groups from the ledger plus UNI2 rank edges, then size the pool.

Two edge sources, deliberately different in kind:

* ledger confirmed twin pairs -- ground truth, already adjudicated by the user,
  so honour them rather than hope the embedding rediscovers them (it only finds
  68% as mutual-NN);
* UNI2 rank edges -- to catch twins nobody has looked at yet.

Adjudicated not-twin pairs are removed at the end so a user "no" always beats a
model "yes". Groups are connected components: everything in a group shares a
split, and every group with one member is a val/test candidate.

Absolute cosine is not used anywhere. At 4683 slides the median top-1 is 0.977,
above the twin median measured in a small pool, so the scale means nothing.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
NEED = 940  # val + test, 10% each


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=1, help="group if partner is within top-K of either slide")
    ap.add_argument("--mutual-k", type=int, default=0, help="also group if both slides are in each other's top-K")
    args = ap.parse_args()

    d = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    sim = d["vecs"] @ d["vecs"].T
    np.fill_diagonal(sim, -1.0)
    n = len(ids)

    topk = np.argpartition(-sim, args.rank, axis=1)[:, : args.rank]
    pos, neg = load_labels()
    negset = {tuple(sorted(p)) for p in neg}

    edges = {}
    for a, b in pos:
        if a in idx and b in idx:
            edges[tuple(sorted([a, b]))] = "ledger_twin"
    n_ledger = len(edges)
    for i in range(n):
        for j in topk[i]:
            key = tuple(sorted([ids[i], ids[int(j)]]))
            if key not in negset:
                edges.setdefault(key, f"uni2_rank{args.rank}")

    if args.mutual_k:
        near = np.argpartition(-sim, args.mutual_k, axis=1)[:, : args.mutual_k]
        member = [set(map(int, r)) for r in near]
        for i in range(n):
            for j in member[i]:
                if i in member[j]:
                    key = tuple(sorted([ids[i], ids[j]]))
                    if key not in negset:
                        edges.setdefault(key, f"uni2_mutual{args.mutual_k}")

    dsu = DSU(ids)
    for a, b in edges:
        dsu.union(a, b)
    root = {i: dsu.find(i) for i in ids}
    groups = pd.Series(root)
    sizes = groups.value_counts()

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    out = pd.DataFrame({"image_id": ids})
    out["group_id"] = out.image_id.map({i: sorted(set(root.values())).index(root[i]) for i in ids})
    out["group_size"] = out.image_id.map(lambda i: int(sizes[root[i]]))
    out["isup_grade"] = out.image_id.map(meta.isup_grade)
    tag = f"rank{args.rank}" + (f"_mutual{args.mutual_k}" if args.mutual_k else "")
    out.sort_values(["group_size", "group_id"], ascending=[False, True]).to_csv(
        G / f"slide_groups_{tag}.csv", index=False
    )

    free = out[out.group_size == 1]
    print(f"edges: {len(edges)} ({n_ledger} from ledger, {len(edges)-n_ledger} new from UNI2 rank-{args.rank})")
    print(f"groups: {len(sizes)} | largest {int(sizes.max())} | singletons {len(free)}")

    print(f"\n{'ISUP':>5} {'free':>6} {'total':>6} {'free %':>8} {'need':>6} {'ok':>4}")
    ok = True
    share = out.isup_grade.value_counts().sort_index()
    for k in share.index:
        have = int((free.isup_grade == k).sum())
        need = int(round(NEED * share[k] / n))
        good = have >= need
        ok &= good
        print(f"{k:5d} {have:6d} {share[k]:6d} {have/share[k]:8.1%} {need:6d} {'yes' if good else 'NO':>4}")

    split_twins = sum(1 for a, b in pos if a in idx and b in idx and root[a] != root[b])
    report = {
        "rank": args.rank,
        "n_edges": len(edges),
        "n_groups": int(len(sizes)),
        "largest_group": int(sizes.max()),
        "singletons": int(len(free)),
        "confirmed_twins_split_across_groups": split_twins,
        "enough_per_grade_for_val_test": bool(ok),
    }
    (G / f"group_report_{tag}.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
