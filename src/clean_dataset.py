"""Run quality checks on Radboud PANDA masks and build a verified clean training set.

Checks:
  1. Grade consistency — every cancer grade in gleason_score has >0 mask pixels
  2. Empty mask — <1% labeled (non-background) tissue
  3. File integrity — mask fails to open via OpenSlide
  4. Benign consistency — negative gleason but cancer pixels in mask

Also removes slides from the 6th-place noisy-label exclusion list
(data/excluded_slide_ids.txt).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
MASKS_DIR = DATA / "masks"
RADBOUD_MASKS_CSV = DATA / "radboud_slides_with_masks.csv"
TRAIN_RADBOUD_CSV = DATA / "train_radboud.csv"
EXCLUDED_TXT = DATA / "excluded_slide_ids.txt"
CLEAN_CSV = DATA / "radboud_clean.csv"
FLAGS_CSV = OUTPUTS / "clean_dataset_flags.csv"
CHECKPOINT_JSON = OUTPUTS / "clean_dataset_checkpoint.json"

COMPETITION = "prostate-cancer-grade-assessment"
KAGGLE = Path(sys.executable).parent / "Scripts" / "kaggle.exe"

TISSUE_RATIO_MIN = 0.01



def derive_grade_from_mask(mask_array: np.ndarray) -> tuple[int | None, int | None]:
    g3_pixels = int((mask_array == 3).sum())
    g4_pixels = int((mask_array == 4).sum())
    g5_pixels = int((mask_array == 5).sum())

    grades = sorted([(g3_pixels, 3), (g4_pixels, 4), (g5_pixels, 5)], reverse=True)
    primary = grades[0][1] if grades[0][0] > 0 else None
    secondary = grades[1][1] if grades[1][0] > 0 else None
    return primary, secondary


def mask_derived_gleason_str(mask_array: np.ndarray) -> str:
    primary, secondary = derive_grade_from_mask(mask_array)
    if primary is None:
        return "negative"
    if secondary is None:
        return str(primary)
    return f"{primary}+{secondary}"


def gleason_grades_in_score(gleason_score: str) -> set[int]:
    if gleason_score == "negative":
        return set()
    return {int(grade) for grade in gleason_score.split("+")}


def check_grade_consistency(mask_array: np.ndarray, gleason_score: str) -> bool:
    """Pass if every cancer grade named in gleason_score has pixels in the mask."""
    for grade in gleason_grades_in_score(gleason_score):
        if int((mask_array == grade).sum()) == 0:
            return False
    return True


def check_mask_completeness(mask_array: np.ndarray) -> bool:
    labeled_pixels = int((mask_array > 0).sum())
    tissue_ratio = labeled_pixels / mask_array.size
    return tissue_ratio >= TISSUE_RATIO_MIN


def check_benign_consistency(mask_array: np.ndarray, gleason_score: str) -> bool:
    has_cancer = bool((mask_array >= 3).any())
    if gleason_score == "negative" and has_cancer:
        return False
    return True


def read_mask_labels(mask_path: Path, level: int = 0) -> np.ndarray:
    """Read mask label array at the given pyramid level (default 0 = full resolution)."""
    slide = openslide.OpenSlide(str(mask_path))
    try:
        width, height = slide.level_dimensions[level]
        arr = np.array(slide.read_region((0, 0), level, (width, height)))
    finally:
        slide.close()
    return arr[:, :, 0] if arr.ndim == 3 else arr


def check_file_integrity(mask_path: Path) -> tuple[bool, np.ndarray | None, str | None]:
    try:
        slide = openslide.OpenSlide(str(mask_path))
        try:
            slide.get_thumbnail((256, 256))
            width, height = slide.level_dimensions[0]
            arr = np.array(slide.read_region((0, 0), 0, (width, height)))
        finally:
            slide.close()
        labels = arr[:, :, 0] if arr.ndim == 3 else arr
        return True, labels, None
    except Exception as exc:
        return False, None, str(exc)


def download_mask(image_id: str) -> Path | None:
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MASKS_DIR / f"{image_id}_mask.tiff"
    if dest.exists():
        return dest

    relative = f"train_label_masks/{image_id}_mask.tiff"
    tmp_dir = MASKS_DIR / "_downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            str(KAGGLE),
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            relative,
            "-p",
            str(tmp_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    basename = Path(relative).name
    downloaded = tmp_dir / basename
    zip_path = tmp_dir / f"{basename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        zip_path.unlink()

    if not downloaded.exists():
        return None

    downloaded.replace(dest)
    return dest


def load_excluded_ids() -> set[str]:
    if not EXCLUDED_TXT.exists():
        raise FileNotFoundError(
            f"Missing {EXCLUDED_TXT}. Run src/build_clean_radboud.py first."
        )
    return {line.strip() for line in EXCLUDED_TXT.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_checkpoint() -> dict[str, dict]:
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(results: dict[str, dict]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")


def evaluate_slide(
    image_id: str,
    gleason_score: str,
    *,
    local_only: bool,
) -> dict:
    mask_path = MASKS_DIR / f"{image_id}_mask.tiff"
    if not mask_path.exists() and not local_only:
        downloaded = download_mask(image_id)
        if downloaded is None:
            return {
                "image_id": image_id,
                "gleason_score": gleason_score,
                "mask_path": "",
                "mask_derived_gleason": "",
                "tissue_ratio": None,
                "integrity_ok": False,
                "grade_consistent": None,
                "mask_complete": None,
                "benign_consistent": None,
                "error": "download_failed",
            }
        mask_path = downloaded
    elif not mask_path.exists():
        return {
            "image_id": image_id,
            "gleason_score": gleason_score,
            "mask_path": "",
            "mask_derived_gleason": "",
            "tissue_ratio": None,
            "integrity_ok": False,
            "grade_consistent": None,
            "mask_complete": None,
            "benign_consistent": None,
            "error": "missing_local_mask",
        }

    integrity_ok, mask_array, error = check_file_integrity(mask_path)
    if not integrity_ok or mask_array is None:
        return {
            "image_id": image_id,
            "gleason_score": gleason_score,
            "mask_path": str(mask_path),
            "mask_derived_gleason": "",
            "tissue_ratio": None,
            "integrity_ok": False,
            "grade_consistent": None,
            "mask_complete": None,
            "benign_consistent": None,
            "error": error or "integrity_failed",
        }

    tissue_ratio = float((mask_array > 0).sum()) / mask_array.size
    derived = mask_derived_gleason_str(mask_array)

    return {
        "image_id": image_id,
        "gleason_score": gleason_score,
        "mask_path": str(mask_path),
        "mask_derived_gleason": derived,
        "tissue_ratio": round(tissue_ratio, 6),
        "integrity_ok": True,
        "grade_consistent": check_grade_consistency(mask_array, gleason_score),
        "mask_complete": check_mask_completeness(mask_array),
        "benign_consistent": check_benign_consistency(mask_array, gleason_score),
        "error": "",
    }


def summarize_flags(df: pd.DataFrame, excluded_ids: set[str]) -> None:
    known_bad = df["image_id"].isin(excluded_ids)

    grade_bad = df["grade_consistent"] == False  # noqa: E712
    empty_bad = df["mask_complete"] == False  # noqa: E712
    integrity_bad = df["integrity_ok"] == False  # noqa: E712
    benign_bad = df["benign_consistent"] == False  # noqa: E712
    qc_bad = grade_bad | empty_bad | integrity_bad | benign_bad
    any_bad = known_bad | qc_bad

    print()
    print("=" * 60)
    print("QUALITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Slides evaluated:              {len(df)}")
    print(f"Known noisy (exclusion list):  {known_bad.sum()} flagged")
    print(f"Check 1 — Grade mismatch:      {grade_bad.sum()} flagged")
    print(f"Check 2 — Empty/near-empty:    {empty_bad.sum()} flagged")
    print(f"Check 3 — Corrupted/unopenable:{integrity_bad.sum()} flagged")
    print(f"Check 4 — Benign + cancer px:  {benign_bad.sum()} flagged")
    print(f"Any QC check failed:           {qc_bad.sum()} flagged")
    print(f"Total removed (known + QC):    {any_bad.sum()} flagged")
    print(f"Final clean slides:            {(~any_bad).sum()}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Radboud PANDA mask dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N slides (for testing)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only process masks already in data/masks/ (no Kaggle download)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/clean_dataset_checkpoint.json",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between Kaggle downloads",
    )
    args = parser.parse_args()

    radboud_masks = pd.read_csv(RADBOUD_MASKS_CSV)
    metadata = pd.read_csv(TRAIN_RADBOUD_CSV)
    slides = radboud_masks.merge(metadata, on="image_id", how="left")

    if args.limit:
        slides = slides.head(args.limit)

    excluded_ids = load_excluded_ids()
    checkpoint = load_checkpoint() if args.resume else {}
    results: dict[str, dict] = dict(checkpoint)

    pending = [row for _, row in slides.iterrows() if row["image_id"] not in results]
    print(f"Processing {len(slides)} Radboud slides ({len(pending)} pending)")
    print(f"Known noisy exclusion list: {len(excluded_ids)} IDs")

    for row in tqdm(pending, desc="Checking masks"):
        image_id = row["image_id"]
        gleason_score = row["gleason_score"]
        results[image_id] = evaluate_slide(
            image_id,
            gleason_score,
            local_only=args.local_only,
        )
        save_checkpoint(results)
        if not args.local_only:
            time.sleep(args.sleep)

    flags = pd.DataFrame(results.values())
    flags["known_noisy"] = flags["image_id"].isin(excluded_ids)
    flags["flag_grade"] = flags["grade_consistent"] == False  # noqa: E712
    flags["flag_empty"] = flags["mask_complete"] == False  # noqa: E712
    flags["flag_integrity"] = flags["integrity_ok"] == False  # noqa: E712
    flags["flag_benign"] = flags["benign_consistent"] == False  # noqa: E712
    flags["flag_any_qc"] = flags[["flag_grade", "flag_empty", "flag_integrity", "flag_benign"]].any(axis=1)
    flags["flag_remove"] = flags["known_noisy"] | flags["flag_any_qc"]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    flags.to_csv(FLAGS_CSV, index=False)

    clean = metadata[
        metadata["image_id"].isin(slides["image_id"])
        & ~metadata["image_id"].isin(flags.loc[flags["flag_remove"], "image_id"])
    ].copy()
    clean.to_csv(CLEAN_CSV, index=False)

    summarize_flags(flags, excluded_ids)
    print(f"\nSaved flag report: {FLAGS_CSV}")
    print(f"Saved clean list:  {CLEAN_CSV}")
    print(f"Checkpoint:        {CHECKPOINT_JSON}")
    print(f"\n=== {len(clean)} Radboud slides remain after all exclusions ===")


if __name__ == "__main__":
    main()
