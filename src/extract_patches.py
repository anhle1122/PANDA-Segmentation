"""Extract 512x512 patches from local PANDA slides (Step 2 + class distribution plot)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

from patch_utils import (
    CLASS_NAMES,
    MASKS_DIR,
    PATCH_INDEX_CSV,
    PATCH_SIZE,
    SLIDES_DIR,
    OUTPUTS,
    counts_to_json,
    dominant_class,
    load_selected_ids,
    pixel_counts,
)

PROJECT_OUTPUT_PLOT = OUTPUTS / "class_distribution.png"


def extract_slide_patches(image_id: str) -> list[dict]:
    slide_path = SLIDES_DIR / f"{image_id}.tiff"
    mask_path = MASKS_DIR / f"{image_id}_mask.tiff"
    if not slide_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing slide or mask for {image_id}")

    mask_slide = openslide.OpenSlide(str(mask_path))
    try:
        width, height = mask_slide.dimensions
        rows: list[dict] = []
        for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                mask_arr = np.array(mask_slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))
                labels = mask_arr[:, :, 0] if mask_arr.ndim == 3 else mask_arr
                if np.all(labels == 0):
                    continue
                counts = pixel_counts(labels)
                rows.append(
                    {
                        "image_id": image_id,
                        "x": x,
                        "y": y,
                        "dominant_class": dominant_class(counts),
                        "pixel_counts_per_class": counts_to_json(counts),
                    }
                )
    finally:
        mask_slide.close()
    return rows


def plot_class_distribution(df: pd.DataFrame, out_path) -> None:
    counts = df["dominant_class"].value_counts().sort_index()
    labels = [f"{c}: {CLASS_NAMES.get(c, c)}" for c in counts.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values, color=plt.cm.tab10(np.linspace(0, 1, len(counts))))
    ax.set_title("Patch counts by dominant class (all slides)")
    ax.set_ylabel("Number of patches")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 512x512 patches from local slides")
    parser.add_argument("--image-ids", nargs="*", default=None, help="Optional subset of image IDs")
    args = parser.parse_args()

    image_ids = args.image_ids or load_selected_ids()
    print(f"Extracting patches from {len(image_ids)} slides (patch size {PATCH_SIZE})")

    all_rows: list[dict] = []
    for image_id in tqdm(image_ids, desc="Slides"):
        rows = extract_slide_patches(image_id)
        all_rows.extend(rows)
        dist = pd.Series([r["dominant_class"] for r in rows]).value_counts().sort_index()
        dist_str = ", ".join(f"{CLASS_NAMES.get(int(k), k)}={v}" for k, v in dist.items())
        print(f"  {image_id}: {len(rows)} patches  ({dist_str})")

    df = pd.DataFrame(all_rows)
    PATCH_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATCH_INDEX_CSV, index=False)

    print("\n" + "=" * 60)
    print("PATCH EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total patches: {len(df)}")
    for image_id, grp in df.groupby("image_id"):
        print(f"  {image_id}: {len(grp)}")
    print("\nClass distribution (dominant class per patch):")
    for cls, count in df["dominant_class"].value_counts().sort_index().items():
        print(f"  {cls} ({CLASS_NAMES.get(int(cls), cls)}): {count}")
    print("=" * 60)

    plot_class_distribution(df, PROJECT_OUTPUT_PLOT)
    print(f"\nSaved patch index -> {PATCH_INDEX_CSV}")
    print(f"Saved class plot  -> {PROJECT_OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
