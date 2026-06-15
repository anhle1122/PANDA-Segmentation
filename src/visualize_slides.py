"""Visualize mask-selected PANDA slides: stats + side-by-side PNGs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SLIDES_DIR = DATA_DIR / "slides"
MASKS_DIR = DATA_DIR / "masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_CSV = DATA_DIR / "train_radboud.csv"
SELECTION_JSON = PROJECT_ROOT / "outputs" / "selected_slides.json"

CLASS_NAMES = {
    0: "background",
    1: "stroma",
    2: "benign",
    3: "G3",
    4: "G4",
    5: "G5",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 180, 0),
    3: (255, 255, 0),
    4: (255, 140, 0),
    5: (255, 0, 0),
}

LOADER = "unknown"


def open_image(path: Path):
    global LOADER
    try:
        import openslide

        LOADER = "openslide"
        return openslide.OpenSlide(str(path)), "openslide"
    except Exception as openslide_error:
        try:
            import tifffile

            LOADER = "tifffile"
            return tifffile.imread(path), "tifffile"
        except Exception as tifffile_error:
            raise RuntimeError(
                f"Could not open {path} with openslide ({openslide_error}) "
                f"or tifffile ({tifffile_error})"
            ) from tifffile_error


def close_handle(handle) -> None:
    if hasattr(handle, "close"):
        handle.close()


def get_dimensions_and_levels(handle) -> tuple[tuple[int, int], int]:
    if hasattr(handle, "dimensions"):
        return handle.dimensions, handle.level_count
    arr = np.asarray(handle)
    return (arr.shape[1], arr.shape[0]), 1


def read_mask_labels_full(mask) -> np.ndarray:
    if hasattr(mask, "level_count"):
        w, h = mask.level_dimensions[0]
        arr = np.array(mask.read_region((0, 0), 0, (w, h)))
        return arr[:, :, 0] if arr.ndim == 3 else arr
    arr = np.asarray(mask)
    return arr[:, :, 0] if arr.ndim == 3 else arr


def get_slide_thumbnail(slide, size=(512, 512)) -> np.ndarray:
    if hasattr(slide, "get_thumbnail"):
        return np.array(slide.get_thumbnail(size).convert("RGB"))
    from PIL import Image

    arr = np.asarray(slide)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return np.array(Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.BILINEAR))


def get_mask_thumbnail_labels(mask, size=(512, 512)) -> np.ndarray:
    if hasattr(mask, "get_thumbnail"):
        thumb = np.array(mask.get_thumbnail(size))
        return thumb[:, :, 0] if thumb.ndim == 3 else thumb
    labels = read_mask_labels_full(mask)
    from PIL import Image

    return np.array(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST))


def analyze_labels(labels: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    rows = []
    for value, count in zip(unique, counts):
        label = int(value)
        rows.append(
            {
                "label": label,
                "class": CLASS_NAMES.get(label, f"unknown({label})"),
                "pixels": int(count),
                "percent": 100.0 * count / total,
            }
        )
    return pd.DataFrame(rows)


def colorize_mask(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in CLASS_COLORS.items():
        colored[labels == label] = color
    return colored


def list_slide_ids() -> list[str]:
    if SELECTION_JSON.exists():
        import json

        data = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
        return data.get("selected", [])
    return sorted(p.stem for p in SLIDES_DIR.glob("*.tiff"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_CSV).set_index("image_id")
    slide_ids = list_slide_ids()

    if not slide_ids:
        print("No slides found.")
        sys.exit(1)

    print(f"Using loader: will try openslide first, then tifffile fallback")
    print(f"Found {len(slide_ids)} slides\n")

    summary_rows = []

    for image_id in slide_ids:
        slide_path = SLIDES_DIR / f"{image_id}.tiff"
        mask_path = MASKS_DIR / f"{image_id}_mask.tiff"
        if not slide_path.exists() or not mask_path.exists():
            print(f"Skipping {image_id}: missing slide or mask")
            continue

        gleason = metadata.loc[image_id, "gleason_score"]
        print("=" * 70)
        print(f"Slide: {image_id} | Gleason: {gleason}")
        print("=" * 70)

        slide, slide_loader = open_image(slide_path)
        mask, mask_loader = open_image(mask_path)
        print(f"Loader: {slide_loader}")

        dimensions, levels = get_dimensions_and_levels(slide)
        width, height = dimensions
        print(f"  Dimensions: {width} x {height}")
        print(f"  Zoom levels: {levels}")

        stats_labels = read_mask_labels_full(mask)
        stats_df = analyze_labels(stats_labels)
        classes_present = [int(v) for v in sorted(stats_df["label"].tolist())]

        print("  Mask classes present:")
        for _, row in stats_df.iterrows():
            print(
                f"    {row['label']} ({row['class']}): "
                f"{row['pixels']:,} px ({row['percent']:.2f}%)"
            )

        row = {
            "image_id": image_id,
            "gleason_score": gleason,
            "dimensions": f"{width}x{height}",
            "classes_present": ", ".join(CLASS_NAMES[c] for c in classes_present),
        }
        for c in range(6):
            match = stats_df[stats_df["label"] == c]
            row[f"class_{c}_pct"] = round(float(match["percent"].iloc[0]), 2) if len(match) else 0.0
        summary_rows.append(row)

        slide_thumb = get_slide_thumbnail(slide, size=(512, 512))
        mask_labels = get_mask_thumbnail_labels(mask, size=(512, 512))
        mask_colored = colorize_mask(mask_labels)

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].imshow(slide_thumb)
        axes[0].set_title("WSI thumbnail (512x512)")
        axes[0].axis("off")

        axes[1].imshow(mask_colored)
        axes[1].set_title("Colored mask")
        axes[1].axis("off")

        legend_labels = sorted(set(int(v) for v in np.unique(mask_labels) if int(v) in CLASS_NAMES))
        legend_patches = [
            Patch(facecolor=np.array(CLASS_COLORS[v]) / 255.0, label=f"{v}={CLASS_NAMES[v]}")
            for v in legend_labels
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=min(len(legend_patches), 6),
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.suptitle(f"{image_id} | Gleason {gleason}", fontsize=11, y=0.98)
        fig.subplots_adjust(bottom=0.12, top=0.90)

        out_path = OUTPUT_DIR / f"slide_{image_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}\n")

        close_handle(slide)
        close_handle(mask)

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    summary_path = OUTPUT_DIR / "visualization_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(f"Image loader used: {LOADER}")

    saved = sorted(OUTPUT_DIR.glob("slide_*.png"))
    print(f"\nSaved {len(saved)} visualization(s) to {OUTPUT_DIR}:")
    for path in saved:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
