#!/usr/bin/env python3
"""Rescan twins on currently alive slides (includes prior not-twins).

Partition (no membership overlap):
  1) edges IoU≥0.70 → connected components
     - size 2 → safe
     - size ≥3 → multi clusters
  2) edges 0.30≤IoU<0.70 where NEITHER endpoint is in any ≥0.70 component → lower

Does NOT modify splits. Writes a fresh out-dir + optional galleries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dedupe_slides_shape_isup import (
    FP_PATH,
    META_PATH,
    SPLITS_DIR,
    choose_keep_drop,
    cluster,
    load_patch_counts,
    mutual_nn_pairs,
)
from patch_utils import PROJECT


def load_prior_not_twin_ids(dup_legacy: Path) -> set[str]:
    ids: set[str] = set()
    for p in [
        dup_legacy / "user_marked_not_same_lower_iou.csv",
        dup_legacy / "user_marked_not_same_safe_pairs.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for c in df.columns:
            if "id" in c.lower():
                ids.update(str(x) for x in df[c].dropna().astype(str) if len(str(x)) >= 20)
    return ids


def annotate_pair(a: str, b: str, not_twin: set[str]) -> str:
    na, nb = a in not_twin, b in not_twin
    if na and nb:
        return "both_prior_not_twin"
    if na or nb:
        return "one_prior_not_twin"
    return ""


def build_partitioned(pairs: pd.DataFrame, meta: pd.DataFrame, patch_counts: dict[str, int], not_twin: set[str]):
    meta_idx = meta.set_index("image_id")

    high = pairs[pairs.shape_iou >= 0.70].copy()
    low = pairs[(pairs.shape_iou >= 0.30) & (pairs.shape_iou < 0.70)].copy()

    high_clusters = cluster(high) if len(high) else []
    in_high: set[str] = set()
    for mems in high_clusters:
        in_high.update(mems)

    safe_rows, multi_rows = [], []
    for mems in sorted(high_clusters, key=lambda m: (-len(m), m[0])):
        if len(mems) == 2:
            a, b = mems
            iou = float(
                high[
                    ((high.image_id_a == a) & (high.image_id_b == b))
                    | ((high.image_id_a == b) & (high.image_id_b == a))
                ].shape_iou.max()
            )
            keep, drop = choose_keep_drop(mems, patch_counts, meta)
            drop_id = drop[0]
            safe_rows.append(
                {
                    "shape_iou": round(iou, 4),
                    "keep_id": keep,
                    "drop_id": drop_id,
                    "keep_n_patches": patch_counts.get(keep, 0),
                    "drop_n_patches": patch_counts.get(drop_id, 0),
                    "split_keep": str(meta_idx.loc[keep, "split"]),
                    "split_drop": str(meta_idx.loc[drop_id, "split"]),
                    "cross_split": str(meta_idx.loc[keep, "split"]) != str(meta_idx.loc[drop_id, "split"]),
                    "isup": int(meta_idx.loc[keep, "isup_grade"]),
                    "gleason": str(meta_idx.loc[keep, "gleason_score"]),
                    "image_id_a": a,
                    "image_id_b": b,
                    "split_a": str(meta_idx.loc[a, "split"]),
                    "split_b": str(meta_idx.loc[b, "split"]),
                    "prior_not_twin_flag": annotate_pair(a, b, not_twin),
                }
            )
        else:
            # max high-edge iou in component
            max_iou = 0.0
            for i, a in enumerate(mems):
                for b in mems[i + 1 :]:
                    hit = high[
                        ((high.image_id_a == a) & (high.image_id_b == b))
                        | ((high.image_id_a == b) & (high.image_id_b == a))
                    ]
                    if len(hit):
                        max_iou = max(max_iou, float(hit.shape_iou.max()))
            multi_rows.append(
                {
                    "n_members": len(mems),
                    "max_iou": round(max_iou, 4),
                    "image_ids": ";".join(mems),
                    "n_prior_not_twin": sum(1 for m in mems if m in not_twin),
                }
            )

    # lower: low edges with neither endpoint in any ≥0.70 component
    lower_rows = []
    for r in low.itertuples(index=False):
        a, b = str(r.image_id_a), str(r.image_id_b)
        if a in in_high or b in in_high:
            continue
        keep, drop = choose_keep_drop([a, b], patch_counts, meta)
        drop_id = drop[0]
        lower_rows.append(
            {
                "shape_iou": float(r.shape_iou),
                "keep_id": keep,
                "drop_id": drop_id,
                "keep_n_patches": patch_counts.get(keep, 0),
                "drop_n_patches": patch_counts.get(drop_id, 0),
                "split_keep": str(meta_idx.loc[keep, "split"]),
                "split_drop": str(meta_idx.loc[drop_id, "split"]),
                "cross_split": str(meta_idx.loc[keep, "split"]) != str(meta_idx.loc[drop_id, "split"]),
                "isup": int(meta_idx.loc[keep, "isup_grade"]),
                "gleason": str(meta_idx.loc[keep, "gleason_score"]),
                "image_id_a": a,
                "image_id_b": b,
                "split_a": str(r.split_a),
                "split_b": str(r.split_b),
                "prior_not_twin_flag": annotate_pair(a, b, not_twin),
            }
        )

    safe = pd.DataFrame(safe_rows).sort_values("shape_iou", ascending=False).reset_index(drop=True)
    if len(safe):
        safe.insert(0, "cluster_id", range(1, len(safe) + 1))
    lower = pd.DataFrame(lower_rows).sort_values("shape_iou", ascending=False).reset_index(drop=True)
    multi = pd.DataFrame(multi_rows).sort_values(["n_members", "max_iou"], ascending=[False, False]).reset_index(drop=True)
    if len(multi):
        multi.insert(0, "cluster_id", range(1, len(multi) + 1))

    # overlap checks
    safe_ids = set(safe.keep_id).union(set(safe.drop_id)) if len(safe) else set()
    lower_ids = set(lower.keep_id).union(set(lower.drop_id)) if len(lower) else set()
    multi_ids: set[str] = set()
    if len(multi):
        for ids in multi.image_ids.astype(str):
            multi_ids.update(x for x in ids.split(";") if x)
    overlap = {
        "safe∩lower": len(safe_ids & lower_ids),
        "safe∩multi": len(safe_ids & multi_ids),
        "lower∩multi": len(lower_ids & multi_ids),
    }
    return safe, lower, multi, overlap, in_high


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou-min", type=float, default=0.30)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "outputs" / "docs" / "slide_duplicates_rescan_alive",
    )
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--keep-adjudicated-edges",
        action="store_true",
        help="Re-show pairs the user already judged not-twins (default: hide them)",
    )
    args = ap.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    legacy = PROJECT / "outputs" / "docs" / "slide_duplicates"

    if not FP_PATH.exists():
        raise SystemExit(f"Missing {FP_PATH}")

    alive_sp = pd.read_csv(SPLITS_DIR / "panda_slide_splits.csv", dtype=str)
    alive = set(alive_sp.image_id.astype(str))
    split_map = dict(zip(alive_sp.image_id.astype(str), alive_sp.split.astype(str)))

    # A confirmed twin that is somehow alive again would pollute every gallery,
    # so refuse to scan until the ledger and the splits agree.
    from twin_ledger import adjudicated_pairs, verify as verify_ledger

    ledger = verify_ledger()
    if not ledger["clean"]:
        raise SystemExit(
            "Twin ledger is out of sync with the splits — fix before rescanning:\n"
            + json.dumps(ledger, indent=2)
        )
    not_twin = load_prior_not_twin_ids(legacy)
    not_twin_alive = not_twin & alive
    print(f"Alive={len(alive)} prior_not_twin_alive={len(not_twin_alive)} (still included in scan)", flush=True)

    data = np.load(FP_PATH)
    sils_all = data["sils"]
    meta_all = pd.read_csv(META_PATH)
    assert len(meta_all) == len(sils_all)
    mask = meta_all.image_id.astype(str).isin(alive).to_numpy()
    meta = meta_all.loc[mask].copy().reset_index(drop=True)
    sils = sils_all[mask]
    meta["image_id"] = meta["image_id"].astype(str)
    meta["split"] = meta["image_id"].map(split_map).fillna("train")
    missing_fp = sorted(alive - set(meta.image_id))
    print(f"fingerprints={len(meta)} missing_fp={len(missing_fp)}", flush=True)

    pairs = mutual_nn_pairs(sils, meta, args.iou_min, args.top_k)
    pairs = pairs.sort_values(["cross_split", "shape_iou"], ascending=[False, False])

    # Cut the edges already judged not-twins before clustering, so a rejected
    # edge cannot glue two slides into a cluster the user has to reject again.
    n_suppressed = 0
    if not args.keep_adjudicated_edges:
        judged = adjudicated_pairs()
        edge = pairs.apply(
            lambda r: tuple(sorted([str(r.image_id_a), str(r.image_id_b)])), axis=1
        )
        hit = edge.isin(judged)
        pairs[hit].to_csv(out / "suppressed_already_judged_not_twin_pairs.csv", index=False)
        n_suppressed = int(hit.sum())
        pairs = pairs[~hit].reset_index(drop=True)
        print(f"suppressed {n_suppressed} already-judged not-twin edges", flush=True)

    pairs.to_csv(out / "dedupe_pairs_shape_isup.csv", index=False)
    print(f"pairs={len(pairs)} cross_split={int(pairs.cross_split.sum()) if len(pairs) else 0}", flush=True)

    patch_counts = load_patch_counts()
    safe, lower, multi, overlap, in_high = build_partitioned(pairs, meta, patch_counts, not_twin_alive)
    safe.to_csv(out / "dedupe_safe_pairs_iou70.csv", index=False)
    lower.to_csv(out / "dedupe_lower_iou_pairs_30_70.csv", index=False)
    multi.to_csv(out / "dedupe_multi_clusters_iou70.csv", index=False)

    # flag prior not-twin pairs for easy review
    if len(safe):
        safe[safe.prior_not_twin_flag != ""].to_csv(out / "safe_pairs_hitting_prior_not_twins.csv", index=False)
    if len(lower):
        lower[lower.prior_not_twin_flag != ""].to_csv(out / "lower_pairs_hitting_prior_not_twins.csv", index=False)

    summary = {
        "scope": "all_alive_including_prior_not_twins",
        "n_alive": len(alive),
        "n_with_fingerprints": int(len(meta)),
        "n_missing_fingerprints": len(missing_fp),
        "n_prior_not_twin_alive_included": len(not_twin_alive),
        "n_suppressed_already_judged_edges": n_suppressed,
        "iou_min": args.iou_min,
        "top_k": args.top_k,
        "n_pairs": int(len(pairs)),
        "n_safe_iou70": int(len(safe)),
        "n_lower_30_70_exclusive": int(len(lower)),
        "n_multi_clusters": int(len(multi)),
        "n_in_high_iou_components": len(in_high),
        "membership_overlap": overlap,
        "applied": False,
        "out_dir": str(out),
        "note": "Detect-only. Safe/lower/multi are membership-disjoint by construction.",
    }
    (out / "rescan_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "README.txt").write_text(
        "Alive twin RESCAN (detect only; includes prior not-twins).\n"
        "Buckets are exclusive: safe & multi from IoU≥0.70 components; "
        "lower = 0.30–0.70 edges with neither end in a ≥0.70 component.\n"
        "Open galleries/index.html after --render.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    assert overlap["safe∩lower"] == 0 and overlap["safe∩multi"] == 0 and overlap["lower∩multi"] == 0

    if args.render:
        from render_dedupe_galleries import main as render_main

        render_main(dup_dir=out, out_dir=out / "galleries")


if __name__ == "__main__":
    main()
