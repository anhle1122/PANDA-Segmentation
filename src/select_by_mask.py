"""Select Radboud slides by mask availability and class coverage, then download."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SLIDES_DIR = DATA_DIR / "slides"
MASKS_DIR = DATA_DIR / "masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RADBOUD_CSV = DATA_DIR / "train_radboud.csv"
COMPETITION = "prostate-cancer-grade-assessment"
KAGGLE = Path(sys.executable).parent / "Scripts" / "kaggle.exe"

CLASS_NAMES = {
    0: "background",
    1: "stroma",
    2: "benign",
    3: "G3",
    4: "G4",
    5: "G5",
}

RANDOM_SEED = 42
MAX_CANDIDATES = 40


def list_mask_files_kaggle() -> list[str]:
    cache_path = OUTPUT_DIR / "kaggle_mask_ids.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_ids: list[str] = []
    page_token = None

    while True:
        cmd = [
            str(KAGGLE),
            "competitions",
            "files",
            "-c",
            COMPETITION,
            "-v",
            "--page-size",
            "200",
            "-q",
        ]
        if page_token:
            cmd.extend(["--page-token", page_token])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

        next_token = None
        for line in result.stdout.splitlines():
            if line.startswith("Next Page Token"):
                next_token = line.split("=", 1)[1].strip()
                continue
            if "train_label_masks/" in line and line.endswith("_mask.tiff"):
                match = re.search(r"train_label_masks/([0-9a-f]+)_mask\.tiff", line)
                if match:
                    image_ids.append(match.group(1))

        if not next_token:
            break
        page_token = next_token
        time.sleep(0.5)

    image_ids = sorted(set(image_ids))
    cache_path.write_text(json.dumps(image_ids, indent=2), encoding="utf-8")
    return image_ids


def download_file(relative_path: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(KAGGLE),
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            relative_path,
            "-p",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    basename = Path(relative_path).name
    downloaded = output_dir / basename
    zip_path = output_dir / f"{basename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()
        downloaded = output_dir / basename
    return downloaded


def mask_labels_present(mask_path: Path) -> dict[int, int]:
    slide = openslide.OpenSlide(str(mask_path))
    level = slide.level_count - 1
    w, h = slide.level_dimensions[level]
    arr = np.array(slide.read_region((0, 0), level, (w, h)))
    slide.close()
    labels = arr[:, :, 0]
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def select_slides_for_class_coverage(candidate_ids: list[str]) -> tuple[list[str], dict]:
    rng = random.Random(RANDOM_SEED)
    shuffled = candidate_ids.copy()
    rng.shuffle(shuffled)

    selected: list[str] = []
    per_slide_classes: dict[str, set[int]] = {}
    covered: set[int] = set()

    for image_id in shuffled[:MAX_CANDIDATES]:
        mask_path = download_file(
            f"train_label_masks/{image_id}_mask.tiff", MASKS_DIR / "_candidates"
        )
        counts = mask_labels_present(mask_path)
        classes = set(counts.keys())
        per_slide_classes[image_id] = classes

        adds_new = bool(classes - covered)
        if adds_new or len(selected) < 5:
            selected.append(image_id)
            covered |= classes
            final_mask = MASKS_DIR / f"{image_id}_mask.tiff"
            final_mask.parent.mkdir(parents=True, exist_ok=True)
            mask_path.replace(final_mask)

        if len(selected) >= 5 and covered >= set(range(6)):
            break

    if len(selected) < 5:
        for image_id in shuffled[MAX_CANDIDATES:]:
            if image_id in selected:
                continue
            mask_path = download_file(
                f"train_label_masks/{image_id}_mask.tiff", MASKS_DIR / "_candidates"
            )
            counts = mask_labels_present(mask_path)
            classes = set(counts.keys())
            per_slide_classes[image_id] = classes
            selected.append(image_id)
            covered |= classes
            final_mask = MASKS_DIR / f"{image_id}_mask.tiff"
            mask_path.replace(final_mask)
            if len(selected) >= 5:
                break

    candidate_dir = MASKS_DIR / "_candidates"
    if candidate_dir.exists():
        for leftover in candidate_dir.glob("*"):
            if leftover.name not in {f"{i}_mask.tiff" for i in selected}:
                leftover.unlink(missing_ok=True)
        try:
            candidate_dir.rmdir()
        except OSError:
            pass

    return selected[:5], {k: sorted(v) for k, v in per_slide_classes.items()}


def inspect_selected(selected: list[str]) -> pd.DataFrame:
    rows = []
    for image_id in selected:
        mask_path = MASKS_DIR / f"{image_id}_mask.tiff"
        counts = mask_labels_present(mask_path)
        total = sum(counts.values())
        classes = sorted(counts.keys())
        dominant = max(counts.items(), key=lambda x: x[1])
        rows.append(
            {
                "image_id": image_id,
                "classes_present": ", ".join(f"{c}={CLASS_NAMES[c]}" for c in classes),
                "label_values": classes,
                "dominant_class": f"{CLASS_NAMES[dominant[0]]} ({100*dominant[1]/total:.1f}%)",
                **{f"class_{c}_pct": round(100 * counts.get(c, 0) / total, 2) for c in range(6)},
            }
        )
        print(f"\n{image_id}")
        for label in classes:
            pct = 100 * counts[label] / total
            print(f"  {label} ({CLASS_NAMES[label]}): {counts[label]:,} px ({pct:.2f}%)")
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Listing train_label_masks/ via Kaggle API...")
    kaggle_mask_ids = list_mask_files_kaggle()
    radboud = pd.read_csv(RADBOUD_CSV)
    radboud_ids = set(radboud["image_id"])
    with_masks = sorted(set(kaggle_mask_ids) & radboud_ids)

    mask_list_path = OUTPUT_DIR / "radboud_slides_with_masks.csv"
    pd.DataFrame({"image_id": with_masks}).to_csv(mask_list_path, index=False)
    print(f"  Kaggle mask files: {len(kaggle_mask_ids)}")
    print(f"  Radboud slides with masks: {len(with_masks)}")
    print(f"  Saved: {mask_list_path}")

    print("\nStep 2-3: Selecting 5 slides for class coverage and downloading masks...")
    selected, preview_classes = select_slides_for_class_coverage(with_masks)
    print(f"  Selected: {selected}")

    union_classes = sorted(set().union(*[set(preview_classes.get(i, [])) for i in selected]))
    print(f"  Classes covered across 5 slides: {union_classes}")
    for label in union_classes:
        print(f"    {label} = {CLASS_NAMES[label]}")

    print("\nStep 4: Downloading WSI slides for selected IDs...")
    for image_id in selected:
        download_file(f"train_images/{image_id}.tiff", SLIDES_DIR)
        print(f"  Downloaded slide: {image_id}")

    selection_path = OUTPUT_DIR / "selected_slides.json"
    selection_path.write_text(
        json.dumps({"selected": selected, "classes_by_slide": preview_classes}, indent=2),
        encoding="utf-8",
    )

    print("\nStep 5: Mask inspection report")
    report = inspect_selected(selected)
    report_path = OUTPUT_DIR / "mask_class_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\nSaved report: {report_path}")
    print(f"Saved selection: {selection_path}")


if __name__ == "__main__":
    main()
