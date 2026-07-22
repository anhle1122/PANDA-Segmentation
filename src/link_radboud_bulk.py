"""Symlink matched Radboud slide+mask pairs from bulk PANDA extract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from patch_utils import DATA, MASKS_DIR, SLIDES_DIR

PROJECT = Path(__file__).resolve().parent.parent
RADBOUD_MASKS_CSV = PROJECT / "data" / "radboud_slides_with_masks.csv"


def find_bulk_root(extract_dir: Path) -> Path:
    if (extract_dir / "train_images").is_dir():
        return extract_dir
    for sub in extract_dir.iterdir():
        if sub.is_dir() and (sub / "train_images").is_dir():
            return sub
    raise FileNotFoundError(f"Could not find train_images/ under {extract_dir}")


def link_pairs(bulk_root: Path, *, force: bool) -> None:
    slides_src = bulk_root / "train_images"
    masks_src = bulk_root / "train_label_masks"
    if not masks_src.is_dir():
        raise FileNotFoundError(f"Missing {masks_src}")

    ids = pd.read_csv(RADBOUD_MASKS_CSV)["image_id"].tolist()
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    linked_slides = linked_masks = missing_slides = missing_masks = 0

    for image_id in ids:
        slide_src = slides_src / f"{image_id}.tiff"
        mask_src = masks_src / f"{image_id}_mask.tiff"
        slide_dest = SLIDES_DIR / f"{image_id}.tiff"
        mask_dest = MASKS_DIR / f"{image_id}_mask.tiff"

        if slide_src.exists():
            if force or not slide_dest.exists():
                if slide_dest.exists() or slide_dest.is_symlink():
                    slide_dest.unlink()
                slide_dest.symlink_to(slide_src.resolve())
            linked_slides += 1
        else:
            missing_slides += 1

        if mask_src.exists():
            if force or not mask_dest.exists():
                if mask_dest.exists() or mask_dest.is_symlink():
                    mask_dest.unlink()
                mask_dest.symlink_to(mask_src.resolve())
            linked_masks += 1
        else:
            missing_masks += 1

    both = sum(
        1 for i in ids
        if (SLIDES_DIR / f"{i}.tiff").exists() and (MASKS_DIR / f"{i}_mask.tiff").exists()
    )

    print()
    print("=== Radboud matched pairs (symlinks) ===")
    print(f"  IDs in list:        {len(ids)}")
    print(f"  Slides linked:      {linked_slides}")
    print(f"  Masks linked:       {linked_masks}")
    print(f"  Matched slide+mask: {both}")
    print(f"  Missing slides:     {missing_slides}")
    print(f"  Missing masks:      {missing_masks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", type=Path, default=DATA / "downloads" / "panda_bulk")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    link_pairs(find_bulk_root(args.extract_dir), force=args.force)


if __name__ == "__main__":
    main()
