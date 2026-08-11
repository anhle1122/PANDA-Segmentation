#!/usr/bin/env python3
"""All 4683 slides in; suspects → train; only twin-free slides → val/test.

Suspect = any slide that appears in:
  * a human-confirmed twin pair, or
  * a UNI2 mutual nearest-neighbour pair that is not an adjudicated not-twin.

Everything else is the clean pool. Val/test are drawn only from that pool,
ISUP-stratified. Train gets the rest of the clean pool plus every suspect
(including all 852 previously dropped twin slides).

If a grade is short of a full 10%+10% clean quota, we take every clean slide
of that grade and note the shortfall — we do not pull suspects into val/test.
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
SPLITS = PROJECT / "outputs" / "splits"
FRAC = {"train": 0.8, "val": 0.1, "test": 0.1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="write outputs/splits/panda_*.csv (backs up first)")
    args = ap.parse_args()

    emb = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in emb["ids"]]
    n = len(ids)
    sim = emb["vecs"] @ emb["vecs"].T
    np.fill_diagonal(sim, -1.0)
    nn = sim.argmax(1)
    mutual = {
        tuple(sorted([ids[i], ids[int(nn[i])]]))
        for i in range(n)
        if nn[nn[i]] == i
    }

    pos, neg = load_labels()
    posset = {tuple(sorted(p)) for p in pos}
    negset = {tuple(sorted(p)) for p in neg}

    suspect: set[str] = set()
    for a, b in posset:
        suspect.add(a)
        suspect.add(b)
    for a, b in mutual:
        if tuple(sorted([a, b])) not in negset:
            suspect.add(a)
            suspect.add(b)

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    clean = [i for i in ids if i not in suspect]
    grades = sorted(meta.isup_grade.astype(int).unique())
    total = meta.loc[ids].isup_grade.astype(int).value_counts()

    rng = np.random.default_rng(args.seed)
    assign: dict[str, str] = {i: "train" for i in suspect}  # all suspects → train
    shortfall = {}

    for g in grades:
        pool = [i for i in clean if int(meta.isup_grade[i]) == g]
        rng.shuffle(pool)
        need_val = int(round(FRAC["val"] * total[g]))
        need_test = int(round(FRAC["test"] * total[g]))
        need = need_val + need_test
        if len(pool) < need:
            shortfall[int(g)] = {"need": need, "have": len(pool), "short": need - len(pool)}
            # take all clean; split half/half between val and test
            n_val = len(pool) // 2
            for i in pool[:n_val]:
                assign[i] = "val"
            for i in pool[n_val:]:
                assign[i] = "test"
        else:
            for i in pool[:need_val]:
                assign[i] = "val"
            for i in pool[need_val : need_val + need_test]:
                assign[i] = "test"
            for i in pool[need_val + need_test :]:
                assign[i] = "train"

    out = pd.DataFrame(
        {
            "image_id": ids,
            "isup_grade": [int(meta.isup_grade[i]) for i in ids],
            "split": [assign[i] for i in ids],
            "suspect": [i in suspect for i in ids],
        }
    )
    out_path = G / "clean_eval_split.csv"
    out.to_csv(out_path, index=False)

    print(f"suspects (→ train): {len(suspect)}")
    print(f"clean pool:         {len(clean)}")
    print(f"{'split':>6} {'slides':>7} {'share':>7} " + " ".join(f"{'ISUP'+str(g):>8}" for g in grades))
    for s in ("train", "val", "test"):
        sub = out[out.split == s]
        cells = " ".join(f"{(sub.isup_grade==g).sum()/total[g]:8.1%}" for g in grades)
        print(f"{s:>6} {len(sub):7d} {len(sub)/n:7.1%} {cells}")

    # safety: no confirmed twin pair and no mutual-NN pair entirely in val/test with a member outside train... 
    # stronger: no confirmed twin or mutual edge with either end in val/test
    eval_ids = set(out.loc[out.split != "train", "image_id"])
    leaks_twin = [(a, b) for a, b in posset if a in eval_ids or b in eval_ids]
    leaks_mut = [
        (a, b)
        for a, b in mutual
        if tuple(sorted([a, b])) not in negset and (a in eval_ids or b in eval_ids)
    ]
    print(f"\nconfirmed twin edges touching val/test: {len(leaks_twin)} (want 0)")
    print(f"mutual-NN edges touching val/test:      {len(leaks_mut)} (want 0)")
    if shortfall:
        print("\nISUP shortfalls (clean pool too small; took all clean, did not pull suspects):")
        for g, d in shortfall.items():
            print(f"  ISUP {g}: have {d['have']} clean, needed {d['need']} (−{d['short']})")

    report = {
        "n_slides": n,
        "n_suspect": len(suspect),
        "n_clean": len(clean),
        "train": int((out.split == "train").sum()),
        "val": int((out.split == "val").sum()),
        "test": int((out.split == "test").sum()),
        "confirmed_twin_edges_touching_eval": len(leaks_twin),
        "mutual_nn_edges_touching_eval": len(leaks_mut),
        "shortfall": shortfall,
        "applied_to_live_splits": False,
    }

    if args.apply:
        # backup current live splits, then write patch-level CSVs from pre_dedupe + new slide split
        import shutil
        from datetime import date

        tag = f"pre_clean_eval_split_{date.today().isoformat()}"
        for split in ("train", "val", "test"):
            cur = SPLITS / f"panda_{split}.csv"
            if cur.exists():
                shutil.copy2(cur, SPLITS / f"panda_{split}_{tag}.csv")
        # rebuild patch rows from pre_dedupe masters
        slide_split = out.set_index("image_id")["split"]
        frames = []
        for split in ("train", "val", "test"):
            src = SPLITS / f"panda_{split}_pre_dedupe.csv"
            if not src.exists():
                src = SPLITS / f"panda_{split}.csv"
            frames.append(pd.read_csv(src, dtype={"image_id": str}))
        patches = pd.concat(frames, ignore_index=True).drop_duplicates()
        patches = patches[patches.image_id.isin(slide_split.index)].copy()
        patches["split"] = patches.image_id.map(slide_split)
        for split in ("train", "val", "test"):
            sub = patches[patches.split == split].drop(columns=["split"], errors="ignore")
            # keep original columns only
            dest = SPLITS / f"panda_{split}.csv"
            # if pre_dedupe had no split col, fine
            cols = [c for c in sub.columns if c != "split"]
            sub[cols].to_csv(dest, index=False)
            print(f"wrote {dest}  rows={len(sub)}  slides={sub.image_id.nunique()}")
        report["applied_to_live_splits"] = True
        report["backup_tag"] = tag

    (G / "clean_eval_split_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    print(f"\nwrote {out_path}  (live splits untouched)" if not args.apply else "\napplied to live splits")


if __name__ == "__main__":
    main()
