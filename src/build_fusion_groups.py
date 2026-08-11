#!/usr/bin/env python3
"""Build groups: ledger twins ∪ mutual-NN ∪ (rank 2-5+ gated by shape IoU).

No drops. Reports feasibility for:
  A) groups assigned to train/val/test (80/10/10, ISUP-stratified)
  B) suspects (group size>=2) → train; only singletons → val/test
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"


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
    ap.add_argument("--iou-thr", type=float, default=0.29)
    ap.add_argument("--max-rank", type=int, default=5, help="include UNI2 ranks 2..max_rank")
    ap.add_argument("--min-cos", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in emb["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    n = len(ids)
    sim = emb["vecs"] @ emb["vecs"].T
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)
    nn = sim.argmax(1)
    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    pos, neg = load_labels()
    posset = {tuple(sorted(p)) for p in pos}
    negset = {tuple(sorted(p)) for p in neg}

    mutual = {
        tuple(sorted([ids[i], ids[int(nn[i])]]))
        for i in range(n)
        if nn[nn[i]] == i
    }

    def best_rank(a, b):
        i, j = idx[a], idx[b]
        ra = int(np.where(order[i] == j)[0][0]) + 1
        rb = int(np.where(order[j] == i)[0][0]) + 1
        return min(ra, rb), float(sim[i, j])

    # candidate rank2..max edges (same ISUP, cos floor); need shape IoU
    cand = {}
    for i, sid in enumerate(ids):
        for r in range(1, args.max_rank):
            j = int(order[i, r])
            oid = ids[j]
            key = tuple(sorted([sid, oid]))
            if key in posset or key in negset or key in mutual:
                continue
            if int(meta.isup_grade[sid]) != int(meta.isup_grade[oid]):
                continue
            br, cos = best_rank(sid, oid)
            if not (2 <= br <= args.max_rank) or cos < args.min_cos:
                continue
            cand[key] = cos

    print(f"rank2-{args.max_rank} same-ISUP candidates (excl ledger/mutual): {len(cand)}")

    # load shape IoU for candidates + known pairs
    need = set(cand) | posset | negset | mutual
    shape: dict[tuple, float] = {}
    for chunk in pd.read_csv(
        G / "all_pairs_grade_agnostic.csv",
        dtype={"image_id_a": str, "image_id_b": str},
        chunksize=400_000,
    ):
        for a, b, iou in zip(
            chunk.image_id_a.to_numpy(),
            chunk.image_id_b.to_numpy(),
            chunk.shape_iou.to_numpy(dtype=float),
        ):
            key = tuple(sorted([a, b]))
            if key in need:
                shape[key] = float(iou)
    print(f"shape IoU found for {len(shape)}/{len(need)} needed pairs")

    # how many cand pass IoU gate
    passed = {k: c for k, c in cand.items() if shape.get(k, -1.0) >= args.iou_thr}
    print(f"pass IoU>={args.iou_thr}: {len(passed)} / {len(cand)}")

    # recall on confirmed twins at rank 2..max
    miss = []
    for a, b in posset:
        if a not in idx or b not in idx:
            continue
        br, cos = best_rank(a, b)
        if 2 <= br <= args.max_rank:
            miss.append((a, b, br, cos, shape.get(tuple(sorted([a, b])), np.nan)))
    miss_df = pd.DataFrame(miss, columns=["a", "b", "rank", "cos", "iou"])
    if len(miss_df):
        caught = (miss_df.iou >= args.iou_thr).sum()
        print(
            f"known twins at rank2-{args.max_rank}: {len(miss_df)}; "
            f"caught by IoU gate: {caught} ({caught/len(miss_df):.0%})"
        )

    # build groups
    dsu = DSU(ids)
    n_ledger = n_mut = n_fuse = 0
    for a, b in posset:
        if a in idx and b in idx:
            dsu.union(a, b)
            n_ledger += 1
    for a, b in mutual:
        if tuple(sorted([a, b])) in negset:
            continue
        dsu.union(a, b)
        n_mut += 1
    for a, b in passed:
        dsu.union(a, b)
        n_fuse += 1

    root = {i: dsu.find(i) for i in ids}
    sizes = Counter(root.values())
    gid_map = {r: k for k, r in enumerate(sorted(sizes))}
    out = pd.DataFrame(
        {
            "image_id": ids,
            "isup_grade": [int(meta.isup_grade[i]) for i in ids],
            "group_id": [gid_map[root[i]] for i in ids],
        }
    )
    out["group_size"] = out.group_id.map(lambda g: int(sizes[sorted(sizes)[g]]))
    # fix group_size via merge
    sz = out.groupby("group_id").size().rename("group_size")
    out = out.drop(columns=["group_size"]).merge(sz, on="group_id")
    tag = f"fusion_iou{args.iou_thr:.2f}_rank2to{args.max_rank}"
    out.to_csv(G / f"slide_groups_{tag}.csv", index=False)

    print(
        f"\ngroups: {len(sizes)} | largest {max(sizes.values())} | "
        f"singletons {(out.group_size==1).sum()} | "
        f"edges ledger={n_ledger} mutual={n_mut} fusion={n_fuse}"
    )

    # --- Option B feasibility: singletons only in val/test ---
    NEED = int(round(0.2 * n))
    clean = out[out.group_size == 1]
    print(f"\n=== Option B (singletons → val/test) ===")
    print(f"clean pool: {len(clean)}  need ~{NEED} for 10+10")
    total = out.isup_grade.value_counts().sort_index()
    ok_b = True
    for g, tot in total.items():
        have = int((clean.isup_grade == g).sum())
        need = int(round(0.2 * tot))
        flag = "OK" if have >= need else "SHORT"
        if have < need:
            ok_b = False
        print(f"  ISUP {g}: clean {have:4d} / need {need:3d}  {flag}")
    print("exact 80/10/10 feasible:" , ok_b)

    # --- Option A: assign whole groups stratified ---
    print(f"\n=== Option A (groups move as units, 80/10/10) ===")
    frac = {"train": 0.8, "val": 0.1, "test": 0.1}
    quota = {s: {g: frac[s] * total[g] for g in total.index} for s in frac}
    have = {s: {g: 0.0 for g in total.index} for s in frac}
    rng = np.random.default_rng(args.seed)
    blocks = [(gid, sub) for gid, sub in out.groupby("group_id")]
    rng.shuffle(blocks)
    # pin very large groups to train to avoid dominating eval
    assign = {}
    for gid, sub in blocks:
        if len(sub) > 8:
            assign[gid] = "train"
            for g, c in sub.isup_grade.value_counts().items():
                have["train"][g] += c
    for gid, sub in blocks:
        if gid in assign:
            continue
        counts = sub.isup_grade.value_counts()
        best, best_score = None, None
        for s in ("val", "test", "train"):
            score = min(
                (quota[s][g] - have[s][g]) / max(quota[s][g], 1e-9) for g in counts.index
            )
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[gid] = best
        for g, c in counts.items():
            have[best][g] += c
    out["split"] = out.group_id.map(assign)
    print(f"{'split':>6} {'slides':>7} {'share':>7} " + " ".join(f"{'ISUP'+str(g):>8}" for g in total.index))
    for s in ("train", "val", "test"):
        sub = out[out.split == s]
        cells = " ".join(f"{(sub.isup_grade==g).sum()/total[g]:8.1%}" for g in total.index)
        print(f"{s:>6} {len(sub):7d} {len(sub)/n:7.1%} {cells}")
    # leak check
    root_split = out.set_index("image_id")
    leaks = sum(
        1
        for a, b in posset
        if a in root_split.index
        and b in root_split.index
        and root_split.split[a] != root_split.split[b]
    )
    print(f"confirmed twin pairs split across splits: {leaks}")
    out.to_csv(G / f"grouped_split_{tag}.csv", index=False)

    # Option B actual assignment
    print(f"\n=== Option B assignment ===")
    assign_b = {i: "train" for i in out.loc[out.group_size > 1, "image_id"]}
    shortfall = {}
    for g in total.index:
        pool = out[(out.group_size == 1) & (out.isup_grade == g)].image_id.tolist()
        rng.shuffle(pool)
        nv = int(round(0.1 * total[g]))
        nt = int(round(0.1 * total[g]))
        if len(pool) < nv + nt:
            shortfall[int(g)] = {"have": len(pool), "need": nv + nt}
            n_val = len(pool) // 2
            for i in pool[:n_val]:
                assign_b[i] = "val"
            for i in pool[n_val:]:
                assign_b[i] = "test"
        else:
            for i in pool[:nv]:
                assign_b[i] = "val"
            for i in pool[nv : nv + nt]:
                assign_b[i] = "test"
            for i in pool[nv + nt :]:
                assign_b[i] = "train"
    out_b = out.copy()
    out_b["split"] = out_b.image_id.map(assign_b)
    print(f"{'split':>6} {'slides':>7} {'share':>7}")
    for s in ("train", "val", "test"):
        sub = out_b[out_b.split == s]
        print(f"{s:>6} {len(sub):7d} {len(sub)/n:7.1%}")
    if shortfall:
        print("shortfalls:", shortfall)
    else:
        print("no shortfalls — exact 80/10/10 per grade achievable")
    out_b.to_csv(G / f"clean_eval_split_{tag}.csv", index=False)

    report = {
        "n_slides": n,
        "iou_thr": args.iou_thr,
        "max_rank": args.max_rank,
        "n_groups": len(sizes),
        "largest_group": int(max(sizes.values())),
        "singletons": int((out.group_size == 1).sum()),
        "fusion_edges": n_fuse,
        "option_b_exact_801010": not bool(shortfall),
        "option_b_shortfall": shortfall,
        "confirmed_twin_leaks_option_a": leaks,
    }
    (G / f"fusion_report_{tag}.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
