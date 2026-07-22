"""Create stratified 80/10/10 slide splits and patch-level CSVs for baseline training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from patch_utils import PROJECT
from slide_list import load_clean_slide_ids

DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
SPLITS_DIR = OUTPUTS / "splits"
CLEAN_CSV = DATA / "radboud_clean.csv"
FILTER_DIR = OUTPUTS / "pen_filter_v33"
RULES_TAG = "v33"


def collect_kept_patches() -> pd.DataFrame:
    rows: list[dict] = []
    for sid in load_clean_slide_ids():
        path = FILTER_DIR / f"pen_filter_{RULES_TAG}_{sid[:12]}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        kept = df[df[f"{RULES_TAG}_action"] == "keep"]
        for r in kept.itertuples():
            rows.append({"image_id": sid, "x": int(r.x), "y": int(r.y)})
    return pd.DataFrame(rows)


def stratified_slide_split(
    slides: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 by slide_id, stratified on isup_grade."""
    train_slides, temp_slides = train_test_split(
        slides,
        test_size=0.2,
        random_state=seed,
        stratify=slides["isup_grade"],
    )
    val_slides, test_slides = train_test_split(
        temp_slides,
        test_size=0.5,
        random_state=seed,
        stratify=temp_slides["isup_grade"],
    )
    return train_slides, val_slides, test_slides


def patches_for_slides(patch_df: pd.DataFrame, slide_ids: set[str]) -> pd.DataFrame:
    return patch_df[patch_df["image_id"].astype(str).isin(slide_ids)].copy()


def print_split_summary(name: str, slides: pd.DataFrame, patches: pd.DataFrame) -> None:
    print(f"\n{name}:")
    print(f"  slides:  {len(slides)}")
    print(f"  patches: {len(patches)}")
    print("  ISUP distribution:")
    for grade, count in slides["isup_grade"].value_counts().sort_index().items():
        print(f"    grade {grade}: {count} slides")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files")
    args = parser.parse_args()

    out_files = {
        "train": SPLITS_DIR / "panda_train.csv",
        "val": SPLITS_DIR / "panda_val.csv",
        "test": SPLITS_DIR / "panda_test.csv",
    }
    if not args.force and all(p.exists() for p in out_files.values()):
        print("Split files already exist — skipping (use --force to rebuild)")
        for name, path in out_files.items():
            df = pd.read_csv(path)
            print(f"  {name}: {len(df)} patches, {df['image_id'].nunique()} slides")
        return

    patch_df = collect_kept_patches()
    slides = pd.read_csv(CLEAN_CSV)[["image_id", "isup_grade"]].astype({"image_id": str})
    slides = slides[slides["image_id"].isin(patch_df["image_id"].astype(str))].copy()

    train_s, val_s, test_s = stratified_slide_split(slides, seed=args.seed)
    train_ids = set(train_s["image_id"])
    val_ids = set(val_s["image_id"])
    test_ids = set(test_s["image_id"])

    train_p = patches_for_slides(patch_df, train_ids)
    val_p = patches_for_slides(patch_df, val_ids)
    test_p = patches_for_slides(patch_df, test_ids)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_p.to_csv(out_files["train"], index=False)
    val_p.to_csv(out_files["val"], index=False)
    test_p.to_csv(out_files["test"], index=False)

    slide_manifest = SPLITS_DIR / "panda_slide_splits.csv"
    pd.concat([
        train_s.assign(split="train"),
        val_s.assign(split="val"),
        test_s.assign(split="test"),
    ]).to_csv(slide_manifest, index=False)

    print(f"Total kept patches: {len(patch_df)} across {patch_df['image_id'].nunique()} slides")
    print_split_summary("train", train_s, train_p)
    print_split_summary("val", val_s, val_p)
    print_split_summary("test", test_s, test_p)
    print(f"\nWrote:\n  {out_files['train']}\n  {out_files['val']}\n  {out_files['test']}")


if __name__ == "__main__":
    main()
