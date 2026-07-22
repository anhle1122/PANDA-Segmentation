"""Run quality checks on Radboud PANDA masks and build a verified clean training set.

A slide is kept in data/radboud_clean.csv only if it passes QC:
  1. Grade consistency — every cancer grade in gleason_score has >0 mask pixels
  2. Mask completeness — >=1% labeled (non-background) tissue
  3. File integrity — mask opens via OpenSlide
  4. Benign consistency — negative gleason has no cancer pixels in mask
  5. Shape match — mask level-0 dimensions match WSI (when slide is local)
  6. Mask file present (and downloadable if not local)

The 6th-place noise_ratio_10 list is NOT used to filter training data. It is
reference-only for post-hoc analysis via src/check_noise_overlap.py after our
own model is trained. QC flags may annotate sixth_place_noise_flag for context.

Outputs (outputs/):
  clean_dataset_flags.csv          — per-slide flags + reason_codes
  clean_dataset_excluded_ids.txt   — all removed image_ids
  clean_dataset_excluded_by_reason.csv — audit trail (image_id, reason, source)
  excluded_<reason>.txt            — one ID list per failure type
  clean_dataset_balance_pre_post.csv — gleason/isup counts before vs after
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

from patch_utils import SLIDES_DIR
from qc_utils import level0_dimensions

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
MASKS_DIR = DATA / "masks"
RADBOUD_MASKS_CSV = DATA / "radboud_slides_with_masks.csv"
TRAIN_RADBOUD_CSV = DATA / "train_radboud.csv"
SIXTH_PLACE_CSV = DATA / "sixth_place_train.csv"
CLEAN_CSV = DATA / "radboud_clean.csv"
FLAGS_CSV = OUTPUTS / "clean_dataset_flags.csv"
EXCLUDED_IDS_TXT = OUTPUTS / "clean_dataset_excluded_ids.txt"
EXCLUDED_BY_REASON_CSV = OUTPUTS / "clean_dataset_excluded_by_reason.csv"
EXCLUSION_AUDIT_CSV = OUTPUTS / "exclusion_audit_log.csv"
MANUAL_EXCLUSIONS_CSV = OUTPUTS / "manual_exclusions.csv"
BALANCE_CSV = OUTPUTS / "clean_dataset_balance_pre_post.csv"
BALANCE_FLAGS_CSV = OUTPUTS / "clean_dataset_balance_flags.csv"
CHECKPOINT_JSON = OUTPUTS / "clean_dataset_checkpoint.json"
SELECTED_JSON = OUTPUTS / "selected_slides.json"

REASON_LABELS = {
    "grade_mismatch": "excluded_grade_mismatch.txt",
    "empty_mask": "excluded_empty_mask.txt",
    "integrity": "excluded_integrity.txt",
    "benign_mismatch": "excluded_benign_mismatch.txt",
    "shape_mismatch": "excluded_shape_mismatch.txt",
    "missing_local_mask": "excluded_missing_mask.txt",
    "missing_local_slide": "excluded_missing_slide.txt",
    "download_failed": "excluded_download_failed.txt",
}

COMPETITION = "prostate-cancer-grade-assessment"


def resolve_kaggle_cli() -> str:
    cli = shutil.which("kaggle")
    if cli:
        return cli
    for candidate in (
        Path(sys.executable).parent / "kaggle",
        Path(sys.executable).parent / "Scripts" / "kaggle.exe",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("kaggle CLI not found")


KAGGLE = resolve_kaggle_cli()

TISSUE_RATIO_MIN = 0.01
# Flag class-balance row if removal % exceeds overall removal % by this margin
BIAS_MARGIN_PCT = 10.0
# Gleason scores / ISUP grades watched as potentially underrepresented
RARE_GLEASON_WATCH = {"5", "5+4", "4+5", "5+5", "3+5", "5+3"}
RARE_ISUP_WATCH = {4, 5}



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


def check_shape_match(mask_path: Path, slide_path: Path) -> tuple[bool | None, str | None]:
    """Return (match, detail). None if slide not available for comparison."""
    if not slide_path.exists():
        return None, "slide_not_local"
    try:
        mask_w, mask_h = level0_dimensions(mask_path)
        slide_w, slide_h = level0_dimensions(slide_path)
    except Exception as exc:
        return False, f"shape_read_error:{exc}"
    if (mask_w, mask_h) != (slide_w, slide_h):
        return False, f"mask={mask_w}x{mask_h} slide={slide_w}x{slide_h}"
    return True, None


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


def load_sixth_place_reference() -> set[str]:
    """Optional reference flags — never used to exclude slides from training."""
    for path in (
        DATA / "sixth_place_noise_flagged_ids.txt",
        DATA / "excluded_slide_ids.txt",
    ):
        if path.exists():
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    if SIXTH_PLACE_CSV.exists():
        df = pd.read_csv(SIXTH_PLACE_CSV)
        if "noise_ratio_10" in df.columns:
            return set(df.loc[df["noise_ratio_10"] == 0, "image_id"].astype(str))
    return set()


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
    slide_path = SLIDES_DIR / f"{image_id}.tiff"
    base = {
        "image_id": image_id,
        "gleason_score": gleason_score,
        "mask_path": "",
        "slide_path": str(slide_path) if slide_path.exists() else "",
        "mask_derived_gleason": "",
        "tissue_ratio": None,
        "integrity_ok": False,
        "grade_consistent": None,
        "mask_complete": None,
        "benign_consistent": None,
        "shape_match": None,
        "shape_detail": "",
        "error": "",
    }

    if not mask_path.exists() and not local_only:
        downloaded = download_mask(image_id)
        if downloaded is None:
            base["error"] = "download_failed"
            return base
        mask_path = downloaded
    elif not mask_path.exists():
        base["error"] = "missing_local_mask"
        return base

    base["mask_path"] = str(mask_path)

    integrity_ok, mask_array, error = check_file_integrity(mask_path)
    if not integrity_ok or mask_array is None:
        base["integrity_ok"] = False
        base["error"] = error or "integrity_failed"
        return base

    tissue_ratio = float((mask_array > 0).sum()) / mask_array.size
    derived = mask_derived_gleason_str(mask_array)
    shape_ok, shape_detail = check_shape_match(mask_path, slide_path)

    base.update(
        {
            "tissue_ratio": round(tissue_ratio, 6),
            "integrity_ok": True,
            "grade_consistent": check_grade_consistency(mask_array, gleason_score),
            "mask_complete": check_mask_completeness(mask_array),
            "benign_consistent": check_benign_consistency(mask_array, gleason_score),
            "shape_match": shape_ok,
            "shape_detail": shape_detail or "",
            "error": "",
        }
    )
    return base


def reason_codes_for_row(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row.get("error") == "missing_local_mask":
        reasons.append("missing_local_mask")
    if row.get("error") == "download_failed":
        reasons.append("download_failed")
    if row.get("shape_match") is None and row.get("error") == "":
        if row.get("shape_detail") == "slide_not_local":
            reasons.append("missing_local_slide")
    if row.get("shape_match") is False:
        reasons.append("shape_mismatch")
    if row.get("grade_consistent") is False:
        reasons.append("grade_mismatch")
    if row.get("mask_complete") is False:
        reasons.append("empty_mask")
    if row.get("integrity_ok") is False:
        reasons.append("integrity")
    if row.get("benign_consistent") is False:
        reasons.append("benign_mismatch")
    return reasons


def apply_flags(flags: pd.DataFrame, sixth_place_ref: set[str]) -> pd.DataFrame:
    flags = flags.copy()
    flags["reason_codes"] = flags.apply(lambda r: "; ".join(reason_codes_for_row(r)), axis=1)
    flags["sixth_place_noise_flag"] = flags["image_id"].isin(sixth_place_ref)
    flags["flag_grade"] = flags["grade_consistent"] == False  # noqa: E712
    flags["flag_empty"] = flags["mask_complete"] == False  # noqa: E712
    flags["flag_integrity"] = flags["integrity_ok"] == False  # noqa: E712
    flags["flag_benign"] = flags["benign_consistent"] == False  # noqa: E712
    flags["flag_shape"] = flags["shape_match"] == False  # noqa: E712
    flags["flag_missing_slide"] = flags["reason_codes"].str.contains("missing_local_slide")
    flags["flag_missing_mask"] = flags["reason_codes"].str.contains("missing_local_mask")
    flags["flag_any_qc"] = flags[
        ["flag_grade", "flag_empty", "flag_integrity", "flag_benign", "flag_shape"]
    ].any(axis=1)
    flags["passes_qc"] = (
        ~flags["flag_any_qc"]
        & ~flags["flag_missing_mask"]
        & ~flags["reason_codes"].str.contains("download_failed")
    )
    flags["flag_remove"] = ~flags["passes_qc"]
    return flags


def load_manual_exclusions() -> pd.DataFrame:
    """Optional manual review file: slide_id, reason [, notes]."""
    if not MANUAL_EXCLUSIONS_CSV.exists():
        return pd.DataFrame(columns=["slide_id", "reason", "notes"])
    df = pd.read_csv(MANUAL_EXCLUSIONS_CSV)
    if "slide_id" not in df.columns or "reason" not in df.columns:
        raise ValueError(
            f"{MANUAL_EXCLUSIONS_CSV} must have columns: slide_id, reason"
        )
    return df


def apply_manual_exclusions(flags: pd.DataFrame) -> pd.DataFrame:
    manual = load_manual_exclusions()
    if manual.empty:
        return flags
    manual_ids = set(manual["slide_id"].astype(str))
    flags = flags.copy()
    flags["flag_manual"] = flags["image_id"].isin(manual_ids)
    flags["flag_remove"] = flags["flag_remove"] | flags["flag_manual"]
    # Append manual reasons to reason_codes
    manual_map = manual.groupby("slide_id")["reason"].apply(
        lambda s: "; ".join(str(x) for x in s)
    ).to_dict()
    for idx, row in flags.iterrows():
        if row["image_id"] in manual_map:
            extra = f"manual:{manual_map[row['image_id']]}"
            existing = row.get("reason_codes", "")
            flags.at[idx, "reason_codes"] = f"{existing}; {extra}".strip("; ")
    return flags


def save_exclusion_artifacts(flags: pd.DataFrame) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    removed = flags.loc[flags["flag_remove"]].copy()
    removed_ids = sorted(removed["image_id"].tolist())
    EXCLUDED_IDS_TXT.write_text(
        "\n".join(removed_ids) + ("\n" if removed_ids else ""), encoding="utf-8"
    )

    audit_rows: list[dict] = []
    per_reason: dict[str, list[str]] = {k: [] for k in REASON_LABELS}

    for _, row in removed.iterrows():
        reasons = reason_codes_for_row(row)
        # Skip automated reasons for purely manual rows (handled below)
        if row.get("flag_manual") and not reasons:
            continue
        for reason in reasons:
            if reason in per_reason:
                per_reason[reason].append(row["image_id"])
            audit_rows.append(
                {
                    "slide_id": row["image_id"],
                    "gleason_score": row.get("gleason_score", ""),
                    "reason": reason,
                    "review_type": "automated",
                    "notes": row.get("reason_codes", ""),
                }
            )

    manual = load_manual_exclusions()
    for _, mrow in manual.iterrows():
        slide_id = str(mrow["slide_id"])
        audit_rows.append(
            {
                "slide_id": slide_id,
                "gleason_score": "",
                "reason": str(mrow["reason"]),
                "review_type": "manual",
                "notes": str(mrow.get("notes", "")),
            }
        )

    audit = pd.DataFrame(audit_rows).drop_duplicates(
        subset=["slide_id", "reason", "review_type"], keep="first"
    )
    audit.to_csv(EXCLUSION_AUDIT_CSV, index=False)
    # Legacy alias for downstream scripts
    audit.rename(columns={"review_type": "source"}).to_csv(EXCLUDED_BY_REASON_CSV, index=False)

    for reason, filename in REASON_LABELS.items():
        ids = sorted(set(per_reason.get(reason, [])))
        path = OUTPUTS / filename
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def save_class_balance(metadata: pd.DataFrame, flags: pd.DataFrame, evaluated_ids: set[str]) -> None:
    meta = metadata[metadata["image_id"].isin(evaluated_ids)].copy()
    keep_ids = set(flags.loc[~flags["flag_remove"], "image_id"])
    n_pre = len(meta)
    n_post = len(meta.loc[meta["image_id"].isin(keep_ids)])
    overall_removed = n_pre - n_post
    overall_pct_removed = round(overall_removed / n_pre * 100, 2) if n_pre else 0.0

    rows: list[dict] = []
    for col in ("gleason_score", "isup_grade"):
        if col not in meta.columns:
            continue
        pre = meta[col].value_counts().sort_index()
        post = meta.loc[meta["image_id"].isin(keep_ids), col].value_counts().sort_index()
        for value in pre.index.union(post.index):
            pre_n = int(pre.get(value, 0))
            post_n = int(post.get(value, 0))
            removed_n = pre_n - post_n
            pct_removed = round(removed_n / pre_n * 100, 2) if pre_n else 0.0
            pct_pre = round(pre_n / n_pre * 100, 2) if n_pre else 0.0

            is_rare_watch = False
            if col == "gleason_score":
                is_rare_watch = str(value) in RARE_GLEASON_WATCH or "5" in str(value).split("+")
            elif col == "isup_grade":
                try:
                    is_rare_watch = int(value) in RARE_ISUP_WATCH
                except (TypeError, ValueError):
                    is_rare_watch = False

            bias_flag = (
                removed_n > 0
                and pct_removed > overall_pct_removed + BIAS_MARGIN_PCT
            )
            rare_bias_flag = is_rare_watch and removed_n > 0 and pct_removed > 0

            rows.append(
                {
                    "column": col,
                    "value": value,
                    "count_pre": pre_n,
                    "count_post": post_n,
                    "pct_pre": pct_pre,
                    "removed": removed_n,
                    "pct_removed": pct_removed,
                    "overall_pct_removed": overall_pct_removed,
                    "bias_flag": bias_flag,
                    "rare_class_watch": is_rare_watch,
                    "rare_bias_red_flag": rare_bias_flag,
                }
            )

    balance = pd.DataFrame(rows)
    balance.to_csv(BALANCE_CSV, index=False)
    flags_df = balance[balance["rare_bias_red_flag"] | balance["bias_flag"]].copy()
    flags_df.to_csv(BALANCE_FLAGS_CSV, index=False)

    print()
    print("=== Class balance (pre vs post exclusion) ===")
    print(f"  Overall: {n_pre} -> {n_post} slides ({overall_pct_removed}% removed)")
    for col in balance["column"].unique():
        print(f"\n  {col}:")
        sub = balance[balance["column"] == col]
        for _, r in sub.iterrows():
            if r["removed"]:
                flag = ""
                if r["rare_bias_red_flag"]:
                    flag = " *** RARE CLASS RED FLAG ***"
                elif r["bias_flag"]:
                    flag = " * bias warning *"
                print(
                    f"    {r['value']}: {int(r['count_pre'])} -> {int(r['count_post'])} "
                    f"(-{int(r['removed'])}, -{r['pct_removed']}% of class, "
                    f"{r['pct_pre']}% of dataset pre){flag}"
                )

    red = balance[balance["rare_bias_red_flag"]]
    if len(red):
        print("\n  RED FLAGS — rare classes disproportionately removed:")
        for _, r in red.iterrows():
            print(
                f"    {r['column']}={r['value']}: {r['pct_removed']}% removed "
                f"(overall {overall_pct_removed}%) — review excluded slides manually"
            )
    else:
        print("\n  No rare-class red flags (Gleason 5 / ISUP 4-5 removal looks proportional).")

    print(f"\n  Saved balance table -> {BALANCE_CSV}")
    print(f"  Saved bias flags   -> {BALANCE_FLAGS_CSV}")


def summarize_flags(df: pd.DataFrame) -> None:
    grade_bad = df["flag_grade"] if "flag_grade" in df else df["grade_consistent"] == False  # noqa: E712
    empty_bad = df["flag_empty"] if "flag_empty" in df else df["mask_complete"] == False  # noqa: E712
    integrity_bad = df["flag_integrity"] if "flag_integrity" in df else df["integrity_ok"] == False  # noqa: E712
    benign_bad = df["flag_benign"] if "flag_benign" in df else df["benign_consistent"] == False  # noqa: E712
    shape_bad = df["flag_shape"] if "flag_shape" in df.columns else False
    qc_bad = grade_bad | empty_bad | integrity_bad | benign_bad | shape_bad
    any_bad = df["flag_remove"] if "flag_remove" in df.columns else qc_bad
    sixth_ref = df["sixth_place_noise_flag"].sum() if "sixth_place_noise_flag" in df.columns else 0

    print()
    print("=" * 60)
    print("QUALITY CHECK SUMMARY (QC-only filtering)")
    print("=" * 60)
    print(f"Slides evaluated:              {len(df)}")
    print(f"6th-place reference flags:     {sixth_ref} (annotation only, not excluded)")
    print()
    print(f"Check 1 — Grade mismatch:      {grade_bad.sum()} flagged")
    print(f"Check 2 — Empty/near-empty:    {empty_bad.sum()} flagged")
    print(f"Check 3 — Corrupted/unopenable:{integrity_bad.sum()} flagged")
    print(f"Check 4 — Benign + cancer px:  {benign_bad.sum()} flagged")
    print(f"Check 5 — Mask/slide shape:    {shape_bad.sum()} flagged")
    print(f"Any QC check failed:           {qc_bad.sum()} flagged")
    passes_qc = df["passes_qc"] if "passes_qc" in df.columns else ~qc_bad
    print(f"Pass all QC checks:            {passes_qc.sum()}")
    print()
    print(f"Total removed (QC only):       {any_bad.sum()}")
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
        "--selected-only",
        action="store_true",
        help="Only process image IDs listed in outputs/selected_slides.json",
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

    if args.selected_only:
        if not SELECTED_JSON.exists():
            raise FileNotFoundError(f"Missing {SELECTED_JSON}")
        selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8")).get("selected", [])
        slides = slides[slides["image_id"].isin(selected)].copy()
        print(f"Selected-only mode: {len(slides)} slides from {SELECTED_JSON.name}")

    if args.limit:
        slides = slides.head(args.limit)

    sixth_place_ref = load_sixth_place_reference()
    checkpoint = load_checkpoint() if args.resume else {}
    results: dict[str, dict] = dict(checkpoint)

    pending = [row for _, row in slides.iterrows() if row["image_id"] not in results]
    print(f"Processing {len(slides)} Radboud slides ({len(pending)} pending)")
    if sixth_place_ref:
        overlap = len(set(slides["image_id"]) & sixth_place_ref)
        print(
            f"6th-place reference flags (NOT excluded): {overlap} slides in this batch"
        )

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
    flags = apply_flags(flags, sixth_place_ref)
    flags = apply_manual_exclusions(flags)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    flags_path = FLAGS_CSV if not args.selected_only else OUTPUTS / "clean_dataset_flags_local.csv"
    flags.to_csv(flags_path, index=False)

    if not args.selected_only:
        save_exclusion_artifacts(flags)
        save_class_balance(metadata, flags, set(slides["image_id"]))

    if args.selected_only:
        passing_ids = set(flags.loc[~flags["flag_remove"], "image_id"])
        print(f"\n(Local demo) {len(passing_ids)}/{len(flags)} selected slides pass all checks.")
    else:
        clean = metadata[
            metadata["image_id"].isin(slides["image_id"])
            & ~metadata["image_id"].isin(flags.loc[flags["flag_remove"], "image_id"])
        ].copy()
        clean.to_csv(CLEAN_CSV, index=False)

    summarize_flags(flags)

    passing = flags[~flags["flag_remove"]]
    if len(passing):
        print("\nSlides passing all checks:")
        for _, row in passing.head(20).iterrows():
            print(f"  PASS  {row['image_id']}  gleason={row['gleason_score']}")
        if len(passing) > 20:
            print(f"  ... and {len(passing) - 20} more")

    failing = flags[flags["flag_remove"]]
    if len(failing):
        print("\nSlides failing one or more checks:")
        for _, row in failing.head(30).iterrows():
            print(
                f"  FAIL  {row['image_id']}  gleason={row['gleason_score']}  "
                f"({row.get('reason_codes', '')})"
            )
        if len(failing) > 30:
            print(f"  ... and {len(failing) - 30} more (see {EXCLUDED_IDS_TXT})")

    highlight_id = "6dfc6b0ab1aa81cb550e52ca291cbb64"
    if highlight_id in set(flags["image_id"]):
        row = flags.loc[flags["image_id"] == highlight_id].iloc[0]
        print(f"\n--- 6dfc slide detail ({highlight_id}) ---")
        print(f"  gleason_score (metadata):     {row['gleason_score']}")
        print(f"  mask_derived_gleason:         {row['mask_derived_gleason']}")
        print(f"  tissue_ratio:                 {row['tissue_ratio']}")
        print(f"  grade_consistent:             {row['grade_consistent']}")
        print(f"  mask_complete:                {row['mask_complete']}")
        print(f"  benign_consistent:            {row['benign_consistent']}")
        print(f"  integrity_ok:                 {row['integrity_ok']}")
        print(f"  overall:                      {'PASS' if not row['flag_remove'] else 'FAIL'}")

    print(f"\nSaved flag report: {flags_path}")
    if not args.selected_only:
        print(f"Saved clean list:       {CLEAN_CSV}")
        print(f"Excluded IDs:           {EXCLUDED_IDS_TXT}")
        print(f"Exclusion audit log:    {EXCLUSION_AUDIT_CSV}")
        print(f"  (columns: slide_id, reason, review_type=automated|manual)")
        print(f"Manual exclusions file: {MANUAL_EXCLUSIONS_CSV} (optional, create to add manual drops)")
        print(f"Class balance:          {BALANCE_CSV}")
        print(f"Balance red flags:      {BALANCE_FLAGS_CSV}")
        print(f"Per-reason ID lists:    outputs/excluded_*.txt")
        print(f"Checkpoint:             {CHECKPOINT_JSON}")
        print(f"\n=== {len(clean)} Radboud slides remain after QC exclusions ===")
        print("Next: python src/scan_pen_marks.py --from-clean")
        print(
            "After training (sufficiently converged checkpoint), pick epoch then run overlap:\n"
            "  python src/suggest_noise_epoch.py --export\n"
            "  python src/check_noise_overlap.py --epoch <N> --top-pct 10"
        )


if __name__ == "__main__":
    main()
