import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

PROJECT = Path(r"C:\Users\LeA14\OneDrive - Cedars-Sinai Health System\Desktop\PANDA_Segmentation")
SLIDES = PROJECT / "data" / "slides"
MASKS = PROJECT / "data" / "masks"
OUTPUT = PROJECT / "outputs"
KAGGLE = Path(r"C:\Users\LeA14\AppData\Local\miniconda3\envs\wsi_seg\Scripts\kaggle.exe")
COMP = "prostate-cancer-grade-assessment"
CLASS = {0: "background", 1: "stroma", 2: "benign", 3: "G3", 4: "G4", 5: "G5"}

selected = sorted(
    p.stem.replace("_mask", "")
    for p in MASKS.glob("*_mask.tiff")
    if p.parent.name != "_candidates"
)


def download_file(relative_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(KAGGLE), "competitions", "download", "-c", COMP, "-f", relative_path, "-p", str(output_dir), "-q"],
        check=True,
    )
    zip_path = output_dir / f"{Path(relative_path).name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        zip_path.unlink()


print("Selected slides:", selected)
print("\nDownloading WSI slides...")
for image_id in selected:
    download_file(f"train_images/{image_id}.tiff", SLIDES)
    print(f"  downloaded {image_id}")

rows = []
for image_id in selected:
    mask = openslide.OpenSlide(str(MASKS / f"{image_id}_mask.tiff"))
    level = mask.level_count - 1
    w, h = mask.level_dimensions[level]
    labels = np.array(mask.read_region((0, 0), level, (w, h)))[:, :, 0]
    mask.close()

    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    classes = [int(v) for v in unique]

    print(f"\n{image_id}")
    row = {
        "image_id": image_id,
        "label_values": classes,
        "classes_present": ", ".join(CLASS[v] for v in classes),
    }
    for label, count in zip(unique, counts):
        pct = 100 * count / total
        print(f"  {int(label)} ({CLASS[int(label)]}): {int(count):,} px ({pct:.2f}%)")
        row[f"class_{int(label)}_pct"] = round(float(pct), 2)

    dominant_label = int(max(zip(unique, counts), key=lambda x: x[1])[0])
    dominant_pct = 100 * counts[list(unique).index(dominant_label)] / total
    row["dominant_class"] = f"{CLASS[dominant_label]} ({dominant_pct:.1f}%)"
    rows.append(row)

OUTPUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT / "mask_class_report.csv", index=False)
(OUTPUT / "selected_slides.json").write_text(json.dumps({"selected": selected}, indent=2), encoding="utf-8")

union = sorted(set().union(*[set(r["label_values"]) for r in rows]))
print("\nUnion of classes across 5 slides:", union)
print("Tissue types covered:", [CLASS[v] for v in union])
print("\nSaved:")
print(" ", OUTPUT / "mask_class_report.csv")
print(" ", OUTPUT / "selected_slides.json")
