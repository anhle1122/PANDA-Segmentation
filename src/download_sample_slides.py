"""Download one Radboud slide + mask per ISUP grade (0-4) from PANDA."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
COMPETITION = "prostate-cancer-grade-assessment"


def download_file(relative_path: str, output_dir: Path) -> None:
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
        raise RuntimeError(
            f"Failed to download {relative_path}:\n{result.stdout}\n{result.stderr}"
        )
    print(result.stdout.strip() or f"Downloaded {relative_path}")


def main() -> None:
    df = pd.read_csv(DATA_DIR / "train_radboud.csv")

    selected = df.groupby("isup_grade").first().reset_index()
    selected = selected[selected["isup_grade"].isin([0, 1, 2, 3, 4])]

    print("Selected slides:")
    print(selected[["image_id", "isup_grade", "gleason_score"]].to_string(index=False))
    print()

    slides_dir = DATA_DIR / "slides"
    masks_dir = DATA_DIR / "masks"

    for _, row in selected.iterrows():
        image_id = row["image_id"]
        isup = row["isup_grade"]

        print(f"Downloading slide ISUP {isup}: {image_id}...")
        download_file(f"train_images/{image_id}.tiff", slides_dir)

        print(f"Downloading mask ISUP {isup}: {image_id}...")
        download_file(f"train_label_masks/{image_id}_mask.tiff", masks_dir)

        print(f"Done: {image_id} (ISUP {isup})\n")

    print("All downloads complete.")
    print(f"Slides: {slides_dir}")
    print(f"Masks:  {masks_dir}")


if __name__ == "__main__":
    main()
