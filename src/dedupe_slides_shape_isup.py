#!/usr/bin/env python3
"""Dedupe WSIs by tissue shape + ISUP/gleason only (stain/pale-dark ignored).

Uses existing silhouette fingerprints from scan_slide_duplicates.py.
Matching rules:
  - same isup_grade AND same gleason_score (hard)
  - silhouette IoU with flip/rot (stain-invariant)
  - mutual top-K nearest neighbor (stops thin-core chain explosions)
  - no pale/dark / RGB brightness checks; dims difference does NOT veto

Keep policy per cluster: keep the slide with the most train patches
(fallback: largest WSI area). Others go on a drop list and are removed
from panda_{train,val,test}.csv + panda_slide_splits.csv.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from patch_utils import PROJECT

DUP_DIR = PROJECT / "outputs" / "docs" / "slide_duplicates"
FP_PATH = DUP_DIR / "fingerprints.npz"
META_PATH = DUP_DIR / "fingerprint_meta.csv"
SPLITS_DIR = PROJECT / "outputs" / "splits"
OUT_PAIRS = DUP_DIR / "dedupe_pairs_shape_isup.csv"
OUT_CLUSTERS = DUP_DIR / "dedupe_clusters_shape_isup.csv"
OUT_DROP = DUP_DIR / "dedupe_drop_ids.txt"
OUT_KEEP = DUP_DIR / "dedupe_keep_ids.txt"
OUT_SUMMARY = DUP_DIR / "dedupe_summary_shape_isup.json"

IOU_MIN = 0.30
TOP_K = 3


def best_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    orients = [
        b,
        np.fliplr(b),
        np.flipud(b),
        np.fliplr(np.flipud(b)),
        np.rot90(b, 1),
        np.rot90(b, 2),
        np.rot90(b, 3),
        np.fliplr(np.rot90(b, 1)),
        np.fliplr(np.rot90(b, 3)),
    ]
    best = 0.0
    aa = int(a.sum())
    for c in orients:
        c = c > 0
        inter = int((a & c).sum())
        union = aa + int(c.sum()) - inter
        if union:
            best = max(best, inter / union)
    return float(best)


def load_patch_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name in ("panda_train.csv", "panda_val.csv", "panda_test.csv"):
        path = SPLITS_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["image_id"])
        vc = df.groupby("image_id").size()
        for sid, n in vc.items():
            counts[str(sid)] += int(n)
    return dict(counts)


def mutual_nn_pairs(sils, meta, iou_min: float, top_k: int) -> pd.DataFrame:
    n = len(meta)
    ids = meta["image_id"].astype(str).tolist()
    isup = meta["isup_grade"].astype(int).tolist()
    gleason = meta["gleason_score"].astype(str).tolist()
    splits = meta["split"].astype(str).tolist()
    widths = meta["width"].astype(int).to_numpy()
    heights = meta["height"].astype(int).to_numpy()

    # group indices by (isup, gleason)
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(isup[i], gleason[i])].append(i)

    # for each slide, top-k neighbors inside its label group
    top: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n)}
    for (_lab, idxs) in tqdm(groups.items(), desc="groups"):
        if len(idxs) < 2:
            continue
        # pairwise within group (gleason groups can be large — OK, IoU is cheap on 128)
        for ii, i in enumerate(idxs):
            scores = []
            for j in idxs:
                if j <= i:
                    continue
                iou = best_iou(sils[i], sils[j])
                if iou >= iou_min:
                    scores.append((iou, j))
                    # also record for j later via symmetric write
            # store candidates for i vs higher j; we'll fill both sides after
            for iou, j in scores:
                top[i].append((iou, j))
                top[j].append((iou, i))

    # keep top_k per slide
    for i in range(n):
        top[i] = sorted(top[i], key=lambda t: -t[0])[:top_k]

    # mutual: j in top_k(i) and i in top_k(j)
    edges = []
    seen = set()
    for i in range(n):
        for iou, j in top[i]:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            if any(jj == i for _, jj in top[j]):
                seen.add((a, b))
                edges.append(
                    {
                        "image_id_a": ids[a],
                        "image_id_b": ids[b],
                        "split_a": splits[a],
                        "split_b": splits[b],
                        "isup": isup[a],
                        "gleason": gleason[a],
                        "dims_a": f"{widths[a]}x{heights[a]}",
                        "dims_b": f"{widths[b]}x{heights[b]}",
                        "shape_iou": round(iou, 4),
                        "cross_split": splits[a] != splits[b],
                    }
                )
    return pd.DataFrame(edges)


def cluster(pairs: pd.DataFrame) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in pairs.itertuples(index=False):
        union(r.image_id_a, r.image_id_b)
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in set(pairs.image_id_a) | set(pairs.image_id_b):
        groups[find(sid)].append(sid)
    return [sorted(v) for v in groups.values() if len(v) >= 2]


def choose_keep_drop(members: list[str], patch_counts: dict[str, int], meta: pd.DataFrame) -> tuple[str, list[str]]:
    area = {
        str(r.image_id): int(r.width) * int(r.height)
        for r in meta.itertuples(index=False)
    }
    def key(sid: str):
        return (patch_counts.get(sid, 0), area.get(sid, 0), sid)

    keep = max(members, key=key)
    drop = [m for m in members if m != keep]
    return keep, drop


def apply_drops(drop_ids: set[str], dry_run: bool) -> dict:
    report = {}
    splits = pd.read_csv(SPLITS_DIR / "panda_slide_splits.csv", dtype=str)
    before = len(splits)
    kept_splits = splits[~splits.image_id.isin(drop_ids)].copy()
    report["slide_splits_before"] = before
    report["slide_splits_after"] = len(kept_splits)
    report["slide_splits_removed"] = before - len(kept_splits)
    if not dry_run:
        # backup once
        bak = SPLITS_DIR / "panda_slide_splits_pre_dedupe.csv"
        if not bak.exists():
            splits.to_csv(bak, index=False)
        kept_splits.to_csv(SPLITS_DIR / "panda_slide_splits.csv", index=False)

    for name in ("panda_train.csv", "panda_val.csv", "panda_test.csv"):
        path = SPLITS_DIR / name
        df = pd.read_csv(path)
        n0, s0 = len(df), df.image_id.nunique()
        out = df[~df.image_id.astype(str).isin(drop_ids)].copy()
        report[name] = {
            "patches_before": int(n0),
            "patches_after": int(len(out)),
            "slides_before": int(s0),
            "slides_after": int(out.image_id.nunique()),
        }
        if not dry_run:
            bak = SPLITS_DIR / f"{path.stem}_pre_dedupe.csv"
            if not bak.exists():
                df.to_csv(bak, index=False)
            out.to_csv(path, index=False)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou-min", type=float, default=IOU_MIN)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--apply", action="store_true", help="Rewrite split CSVs (backs up *_pre_dedupe.csv)")
    ap.add_argument("--dry-run", action="store_true", help="With --apply, only print counts")
    args = ap.parse_args()

    if not FP_PATH.exists():
        raise SystemExit(f"Missing {FP_PATH}; run scan_slide_duplicates.py first")

    data = np.load(FP_PATH)
    sils = data["sils"]
    meta = pd.read_csv(META_PATH)
    assert len(meta) == len(sils)

    print(f"Loaded {len(meta)} fingerprints | iou_min={args.iou_min} top_k={args.top_k}")
    print("Rules: same ISUP+gleason, shape IoU, mutual top-k — ignore pale/dark and dim gaps")

    pairs = mutual_nn_pairs(sils, meta, args.iou_min, args.top_k)
    pairs = pairs.sort_values(["cross_split", "shape_iou"], ascending=[False, False])
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"Mutual-NN pairs: {len(pairs)}  cross-split: {int(pairs.cross_split.sum()) if len(pairs) else 0}")

    patch_counts = load_patch_counts()
    clusters = cluster(pairs) if len(pairs) else []
    rows = []
    drop_ids: list[str] = []
    keep_ids: list[str] = []
    for ci, members in enumerate(sorted(clusters, key=lambda m: -len(m)), 1):
        keep, drop = choose_keep_drop(members, patch_counts, meta)
        keep_ids.append(keep)
        drop_ids.extend(drop)
        sub = meta[meta.image_id.isin(members)]
        splits = sorted(sub.split.unique())
        rows.append(
            {
                "cluster_id": ci,
                "n_members": len(members),
                "keep_id": keep,
                "drop_ids": ";".join(drop),
                "keep_n_patches": patch_counts.get(keep, 0),
                "splits": ";".join(splits),
                "cross_split": len(set(splits)) > 1,
                "isup": int(sub.isup_grade.iloc[0]),
                "gleason": str(sub.gleason_score.iloc[0]),
                "image_ids": ";".join(members),
            }
        )
    cl = pd.DataFrame(rows)
    if len(cl):
        cl.to_csv(OUT_CLUSTERS, index=False)
    OUT_DROP.write_text("\n".join(sorted(set(drop_ids))) + ("\n" if drop_ids else ""))
    OUT_KEEP.write_text("\n".join(sorted(set(keep_ids))) + ("\n" if keep_ids else ""))

    # recover known user pairs
    known = [
        ("48440a60", "4889d110"),
        ("dd11c914", "72e64850"),
        ("e707ef8c", "507ac341"),
        ("32291426a698", "6e6c2361d595"),
        ("48b90b806560", "d1071efa4e88"),
    ]
    recovered = {}
    if len(pairs):
        for a, b in known:
            hit = pairs[
                (pairs.image_id_a.str.startswith(a) & pairs.image_id_b.str.startswith(b))
                | (pairs.image_id_a.str.startswith(b) & pairs.image_id_b.str.startswith(a))
            ]
            recovered[f"{a}/{b}"] = float(hit.shape_iou.iloc[0]) if len(hit) else None

    summary = {
        "iou_min": args.iou_min,
        "top_k": args.top_k,
        "n_pairs": int(len(pairs)),
        "n_pairs_cross_split": int(pairs.cross_split.sum()) if len(pairs) else 0,
        "n_clusters": int(len(cl)),
        "n_clusters_cross_split": int(cl.cross_split.sum()) if len(cl) else 0,
        "n_drop": len(set(drop_ids)),
        "n_keep_representatives": len(set(keep_ids)),
        "known_pairs_recovered": recovered,
        "applied": False,
    }
    print(json.dumps(summary, indent=2))

    if args.apply:
        report = apply_drops(set(drop_ids), dry_run=args.dry_run)
        summary["applied"] = not args.dry_run
        summary["apply_report"] = report
        print("APPLY", "dry-run" if args.dry_run else "WROTE splits", json.dumps(report, indent=2))

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PAIRS}, {OUT_CLUSTERS}, {OUT_DROP}, {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
