#!/usr/bin/env python3
"""Rebalance slide splits to ~80/10/10 with all dup-suspects forced into train.

Val/test are filled only from clean (non-suspect) slides, stratified by ISUP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from patch_utils import PROJECT

DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
SPLITS = PROJECT / "outputs" / "splits"
RNG = np.random.default_rng(20260809)


def collect_suspects(alive: set[str]) -> set[str]:
    suspect: set[str] = set()

    def add_ids(xs):
        for x in xs:
            if pd.isna(x):
                continue
            s = str(x).strip()
            if s in alive:
                suspect.add(s)

    for p in [
        DUP / "dedupe_safe_pairs_iou70.csv",
        DUP / "dedupe_lower_iou_pairs_30_70.csv",
        DUP / "user_marked_not_same_lower_iou.csv",
        DUP / "user_marked_not_same_safe_pairs.csv",
        DUP / "lower_iou_implied_twins_decisions.csv",
        DUP / "lower_iou_twin_drops_clean_keep.csv",
        DUP / "user_multi_C1_35_twin_drops.csv",
        DUP / "user_restore_multi_not_twins.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for c in df.columns:
            if "id" in c.lower():
                add_ids(df[c])

    mc = pd.read_csv(DUP / "dedupe_multi_clusters_iou70.csv")
    for ids in mc.image_ids.astype(str):
        add_ids(ids.split(";"))

    force = DUP / "suspects_forced_train_ids.txt"
    if force.exists():
        add_ids(force.read_text().split())

    pairs = pd.read_csv(DUP / "dedupe_pairs_shape_isup.csv")
    both = pairs.image_id_a.astype(str).isin(alive) & pairs.image_id_b.astype(str).isin(alive)
    add_ids(pairs.loc[both, "image_id_a"])
    add_ids(pairs.loc[both, "image_id_b"])
    return suspect


def stratified_val_test(clean_df: pd.DataFrame, n_val: int, n_test: int, rng: np.random.Generator):
    """Pick val/test from clean slides with approximate ISUP stratification."""
    val, test = [], []
    for _, g in clean_df.groupby("isup"):
        ids = g.image_id.tolist()
        rng.shuffle(ids)
        frac = len(ids) / len(clean_df)
        n_v = min(len(ids), int(round(n_val * frac)))
        n_t = min(len(ids) - n_v, int(round(n_test * frac)))
        val.extend(ids[:n_v])
        test.extend(ids[n_v : n_v + n_t])

    assigned = set(val) | set(test)
    pool = [x for x in clean_df.image_id.tolist() if x not in assigned]
    rng.shuffle(pool)
    while len(val) < n_val and pool:
        val.append(pool.pop())
    while len(test) < n_test and pool:
        test.append(pool.pop())

    # trim overflow back to pool
    rng.shuffle(val)
    rng.shuffle(test)
    if len(val) > n_val:
        pool.extend(val[n_val:])
        val = val[:n_val]
    if len(test) > n_test:
        pool.extend(test[n_test:])
        test = test[:n_test]

    # final top-up if still short (from pool)
    rng.shuffle(pool)
    while len(val) < n_val and pool:
        val.append(pool.pop())
    while len(test) < n_test and pool:
        test.append(pool.pop())

    assert len(val) == n_val and len(test) == n_test
    assert len(set(val) & set(test)) == 0
    return set(val), set(test)


def main() -> None:
    print("loading splits...", flush=True)
    sp = pd.read_csv(SPLITS / "panda_slide_splits.csv", dtype=str)
    tr = pd.read_csv(SPLITS / "panda_train.csv")
    va = pd.read_csv(SPLITS / "panda_val.csv")
    te = pd.read_csv(SPLITS / "panda_test.csv")
    print(f"current {sp.split.value_counts().to_dict()} total={len(sp)}", flush=True)

    for name, df in [
        ("panda_slide_splits", sp),
        ("panda_train", tr),
        ("panda_val", va),
        ("panda_test", te),
    ]:
        bak = SPLITS / f"{name}_pre_rebalance_80_10_10.csv"
        if not bak.exists():
            print(f"backup {bak.name}", flush=True)
            df.to_csv(bak, index=False)

    alive = set(sp.image_id.astype(str))
    print("collecting suspects...", flush=True)
    suspects = collect_suspects(alive)
    clean = sorted(alive - suspects)
    print(f"suspects={len(suspects)} clean={len(clean)}", flush=True)

    n = len(alive)
    n_val = int(round(n * 0.10))
    n_test = int(round(n * 0.10))
    n_train = n - n_val - n_test
    if len(suspects) > n_train:
        raise SystemExit(f"Too many suspects ({len(suspects)}) for train ({n_train})")

    n_clean_train = n_train - len(suspects)
    assert n_clean_train + n_val + n_test == len(clean)

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str})
    isup_map = dict(zip(meta.image_id.astype(str), meta.isup_grade.astype(int)))
    clean_df = pd.DataFrame({"image_id": clean})
    clean_df["isup"] = clean_df.image_id.map(lambda x: isup_map.get(x, -1))

    print("stratified val/test from clean...", flush=True)
    val_ids, test_ids = stratified_val_test(clean_df, n_val, n_test, RNG)
    train_clean = set(clean) - val_ids - test_ids
    assert len(train_clean) == n_clean_train

    train_ids = set(suspects) | train_clean
    assert len(train_ids) == n_train
    assert set(suspects).issubset(train_ids)
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

    sp2 = sp.copy()
    sp2.loc[sp2.image_id.isin(train_ids), "split"] = "train"
    sp2.loc[sp2.image_id.isin(val_ids), "split"] = "val"
    sp2.loc[sp2.image_id.isin(test_ids), "split"] = "test"

    print("rebuilding patch tables...", flush=True)
    all_patches = pd.concat([tr, va, te], ignore_index=True).drop_duplicates(["image_id", "x", "y"])
    have = set(all_patches.image_id.astype(str))
    missing = sorted(alive - have)
    if missing:
        # only pull from the Friday-ish full backup once
        print(f"missing patches for {len(missing)} slides; filling from pre_dedupe", flush=True)
        for name in ("panda_train_pre_dedupe.csv", "panda_val_pre_dedupe.csv", "panda_test_pre_dedupe.csv"):
            df = pd.read_csv(SPLITS / name)
            hit = df[df.image_id.astype(str).isin(missing)]
            if len(hit):
                all_patches = pd.concat([all_patches, hit], ignore_index=True)
        all_patches = all_patches.drop_duplicates(["image_id", "x", "y"])
        missing = sorted(alive - set(all_patches.image_id.astype(str)))
        print(f"still missing {len(missing)}", flush=True)

    tr2 = all_patches[all_patches.image_id.astype(str).isin(train_ids)]
    va2 = all_patches[all_patches.image_id.astype(str).isin(val_ids)]
    te2 = all_patches[all_patches.image_id.astype(str).isin(test_ids)]

    print("writing CSVs...", flush=True)
    sp2.to_csv(SPLITS / "panda_slide_splits.csv", index=False)
    tr2.to_csv(SPLITS / "panda_train.csv", index=False)
    va2.to_csv(SPLITS / "panda_val.csv", index=False)
    te2.to_csv(SPLITS / "panda_test.csv", index=False)

    def isup_mix(ids):
        xs = [isup_map.get(i, -1) for i in ids]
        s = pd.Series(xs).value_counts(normalize=True).sort_index()
        return {int(k): round(float(v) * 100, 2) for k, v in s.items()}

    summary = {
        "total": n,
        "splits": sp2.split.value_counts().to_dict(),
        "pct": {k: round(v / n * 100, 2) for k, v in sp2.split.value_counts().items()},
        "n_suspects_all_train": len(suspects),
        "n_clean_train": n_clean_train,
        "n_clean_val": int(len(val_ids)),
        "n_clean_test": int(len(test_ids)),
        "patches": {"train": int(len(tr2)), "val": int(len(va2)), "test": int(len(te2))},
        "patch_slides": {
            "train": int(tr2.image_id.nunique()),
            "val": int(va2.image_id.nunique()),
            "test": int(te2.image_id.nunique()),
        },
        "isup_pct": {
            "train": isup_mix(train_ids),
            "val": isup_mix(val_ids),
            "test": isup_mix(test_ids),
        },
        "suspect_definition": "curated safe/lower/multi/user lists ∪ alive mutual-NN pair edges",
        "val_test_are_clean_only": True,
        "suspect_leak_val_test": 0,
    }
    leak = (set(suspects) & val_ids) | (set(suspects) & test_ids)
    summary["suspect_leak_val_test"] = len(leak)

    (DUP / "rebalance_80_10_10_summary.json").write_text(json.dumps(summary, indent=2))
    (DUP / "suspects_forced_train_ids.txt").write_text("\n".join(sorted(suspects)) + "\n")
    pd.DataFrame(
        {
            "image_id": sorted(val_ids) + sorted(test_ids),
            "split": ["val"] * len(val_ids) + ["test"] * len(test_ids),
        }
    ).to_csv(DUP / "rebalance_80_10_10_clean_val_test_ids.csv", index=False)

    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
