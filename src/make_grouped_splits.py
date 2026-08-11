#!/usr/bin/env python3
"""Assign whole slide groups to train/val/test, stratified by ISUP grade.

Groups move as units, so a twin can never straddle a split. That replaces
dropping: all 4683 slides stay, duplicates just get confined to one side.

Large groups are pinned to train. They are redundant by construction, and a
40-slide cluster in a 473-slide test set would let one specimen carry 8% of the
score. Train is big enough to absorb them without distorting anything.

The remaining small groups are then assigned greedily, each going to whichever
split is furthest below its per-grade quota.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
SPLITS = PROJECT / "outputs" / "splits"
FRAC = {"train": 0.8, "val": 0.1, "test": 0.1}


def assign_groups(
    df: pd.DataFrame,
    *,
    seed: int,
    max_eval_group: int,
) -> pd.DataFrame:
    grades = sorted(df.isup_grade.unique())
    total = df.isup_grade.value_counts()
    quota = {s: {g: FRAC[s] * total[g] for g in grades} for s in FRAC}
    have = {s: {g: 0.0 for g in grades} for s in FRAC}

    rng = np.random.default_rng(seed)
    blocks = [(gid, sub) for gid, sub in df.groupby("group_id")]
    rng.shuffle(blocks)

    assign: dict = {}
    for gid, sub in [b for b in blocks if len(b[1]) > max_eval_group]:
        assign[gid] = "train"
        for g, c in sub.isup_grade.value_counts().items():
            have["train"][g] += c

    for gid, sub in [b for b in blocks if len(b[1]) <= max_eval_group]:
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

    out = df.copy()
    out["split"] = out.group_id.map(assign)
    return out


def print_summary(df: pd.DataFrame) -> None:
    grades = sorted(df.isup_grade.unique())
    total = df.isup_grade.value_counts()
    print(
        f"{'split':>6} {'slides':>7} {'share':>7} "
        + " ".join(f"{'ISUP'+str(g):>8}" for g in grades)
    )
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        cells = " ".join(f"{(sub.isup_grade==g).sum()/total[g]:8.1%}" for g in grades)
        print(f"{s:>6} {len(sub):7d} {len(sub)/len(df):7.1%} {cells}")

    print(f"\n{'split':>6} {'slides':>7} {'groups':>7} {'largest':>8}")
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        sz = sub.groupby("group_id").size()
        print(f"{s:>6} {len(sub):7d} {len(sz):7d} {int(sz.max()):8d}")


def apply_to_live(df: pd.DataFrame, *, backup_tag: str) -> dict:
    """Backup live patch CSVs, rewrite from pre_dedupe masters + new slide split."""
    for split in ("train", "val", "test"):
        cur = SPLITS / f"panda_{split}.csv"
        if cur.exists():
            shutil.copy2(cur, SPLITS / f"panda_{split}_{backup_tag}.csv")
    slide_csv = SPLITS / "panda_slide_splits.csv"
    if slide_csv.exists():
        shutil.copy2(slide_csv, SPLITS / f"panda_slide_splits_{backup_tag}.csv")

    slide_split = df.set_index("image_id")["split"]
    frames = []
    for split in ("train", "val", "test"):
        src = SPLITS / f"panda_{split}_pre_dedupe.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        frames.append(pd.read_csv(src, dtype={"image_id": str}))
    patches = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["image_id", "x", "y"]
    )
    patches = patches[patches.image_id.isin(slide_split.index)].copy()
    patches["split"] = patches.image_id.map(slide_split)

    stats = {}
    for split in ("train", "val", "test"):
        sub = patches[patches.split == split].drop(columns=["split"], errors="ignore")
        cols = [c for c in ("image_id", "x", "y") if c in sub.columns]
        dest = SPLITS / f"panda_{split}.csv"
        sub[cols].to_csv(dest, index=False)
        stats[split] = {
            "rows": int(len(sub)),
            "slides": int(sub.image_id.nunique()),
        }
        print(f"wrote {dest}  rows={len(sub)}  slides={sub.image_id.nunique()}")

    manifest = df[["image_id", "isup_grade", "split"]].copy()
    if "group_id" in df.columns:
        manifest["group_id"] = df["group_id"]
        manifest["group_size"] = df["group_size"] if "group_size" in df.columns else (
            df.groupby("group_id")["image_id"].transform("size")
        )
    manifest.to_csv(SPLITS / "panda_slide_splits.csv", index=False)
    print(f"wrote {SPLITS / 'panda_slide_splits.csv'}  slides={len(manifest)}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups-csv",
        type=Path,
        default=None,
        help="Groups CSV (default: slide_groups_rank{rank}.csv or fusion)",
    )
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument(
        "--fusion-tag",
        type=str,
        default="fusion_iou0.29_rank2to5",
        help="Used when --groups-csv omitted and --use-fusion is set",
    )
    ap.add_argument("--use-fusion", action="store_true", help="Use fusion groups CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-eval-group",
        type=int,
        default=2,
        help="groups larger than this are pinned to train",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="backup live panda_*.csv then rewrite from pre_dedupe + this split",
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-report", type=Path, default=None)
    args = ap.parse_args()

    if args.groups_csv is not None:
        groups_path = args.groups_csv
        tag = groups_path.stem.replace("slide_groups_", "")
    elif args.use_fusion:
        groups_path = G / f"slide_groups_{args.fusion_tag}.csv"
        tag = args.fusion_tag
    else:
        groups_path = G / f"slide_groups_rank{args.rank}.csv"
        tag = f"rank{args.rank}"

    df = pd.read_csv(groups_path, dtype={"image_id": str})
    if "group_size" not in df.columns:
        df["group_size"] = df.groupby("group_id")["image_id"].transform("size")

    df = assign_groups(df, seed=args.seed, max_eval_group=args.max_eval_group)
    print_summary(df)

    out_csv = args.out_csv or (G / f"grouped_split_{tag}_maxeval{args.max_eval_group}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    root = df.set_index("image_id")
    pos, _ = load_labels()
    known = [(a, b) for a, b in pos if a in root.index and b in root.index]
    leaks = [(a, b) for a, b in known if root.split[a] != root.split[b]]
    crossing = int((df.groupby("group_id").split.nunique() > 1).sum())

    report = {
        "groups_csv": str(groups_path),
        "tag": tag,
        "max_eval_group": args.max_eval_group,
        "seed": args.seed,
        "n_slides": len(df),
        "n_groups": int(df.group_id.nunique()),
        "val_groups": int(df[df.split == "val"].group_id.nunique()),
        "test_groups": int(df[df.split == "test"].group_id.nunique()),
        "largest_group_in_val": int(df[df.split == "val"].groupby("group_id").size().max()),
        "largest_group_in_test": int(df[df.split == "test"].groupby("group_id").size().max()),
        "largest_group_in_train": int(df[df.split == "train"].groupby("group_id").size().max()),
        "confirmed_twin_pairs_checked": len(known),
        "confirmed_twin_pairs_leaking_across_splits": len(leaks),
        "groups_split_across_splits": crossing,
        "val": int((df.split == "val").sum()),
        "test": int((df.split == "test").sum()),
        "train": int((df.split == "train").sum()),
        "applied_to_live_splits": False,
    }

    if args.apply:
        if crossing or leaks:
            raise SystemExit(
                f"Refusing to apply: crossing_groups={crossing} twin_leaks={len(leaks)}"
            )
        backup_tag = f"pre_grouped_{tag}_maxeval{args.max_eval_group}_{date.today().isoformat()}"
        stats = apply_to_live(df, backup_tag=backup_tag)
        report["applied_to_live_splits"] = True
        report["backup_tag"] = backup_tag
        report["live_patch_stats"] = stats

    out_report = args.out_report or (
        G / f"grouped_split_report_{tag}_maxeval{args.max_eval_group}.json"
    )
    out_report.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
