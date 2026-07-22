"""Scan Radboud slides for pen marks; write slide-level flag list.

Detects green, blue, and black ink on WSI thumbnails (see qc_utils.pen_mark_masks).

Usage:
  python src/scan_pen_marks.py
  python src/scan_pen_marks.py --from-clean
  python src/scan_pen_marks.py --from-clean --resume
  python src/scan_pen_marks.py --threshold 0.0001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from patch_utils import OUTPUTS, SLIDES_DIR
from qc_utils import (
    PEN_COLOR_FLAG_FRACTION,
    PEN_MARK_FLAG_FRACTION,
    PEN_MARK_HEURISTIC_VERSION,
    pen_mark_is_flagged,
    slide_pen_mark_fractions,
)

PROJECT = Path(__file__).resolve().parent.parent
RADBOUD_MASKS_CSV = PROJECT / "data" / "radboud_slides_with_masks.csv"
RADBOUD_CLEAN_CSV = PROJECT / "data" / "radboud_clean.csv"
PEN_SLIDES_CSV = OUTPUTS / "pen_mark_slides.csv"
PEN_SLIDE_IDS_TXT = OUTPUTS / "pen_mark_slide_ids.txt"
PEN_SLIDE_IDS_JSON = OUTPUTS / "pen_mark_slide_ids.json"
PEN_CHECKPOINT_JSON = OUTPUTS / "pen_mark_scan_checkpoint.json"

DEFAULT_THRESHOLD = PEN_MARK_FLAG_FRACTION


def load_slide_ids(*, from_clean: bool) -> list[str]:
    if from_clean:
        if not RADBOUD_CLEAN_CSV.exists():
            raise FileNotFoundError(
                f"Missing {RADBOUD_CLEAN_CSV}. Run clean_dataset.py first."
            )
        df = pd.read_csv(RADBOUD_CLEAN_CSV)
    else:
        df = pd.read_csv(RADBOUD_MASKS_CSV)
    return sorted(df["image_id"].unique())


def load_checkpoint() -> dict:
    if PEN_CHECKPOINT_JSON.exists():
        return json.loads(PEN_CHECKPOINT_JSON.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(
    rows: list[dict],
    *,
    from_clean: bool,
    threshold: float,
    thumbnail_edge: int,
) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "heuristic_version": PEN_MARK_HEURISTIC_VERSION,
        "from_clean": from_clean,
        "threshold": threshold,
        "thumbnail_edge": thumbnail_edge,
        "rows": rows,
    }
    PEN_CHECKPOINT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pen_colors_detected(fracs: dict[str, float]) -> str:
    colors = []
    for name in ("green", "blue", "black"):
        if fracs[name] >= PEN_COLOR_FLAG_FRACTION[name]:
            colors.append(name)
    return ",".join(colors)


def scan_one(image_id: str, *, threshold: float, thumbnail_edge: int) -> dict:
    slide_path = SLIDES_DIR / f"{image_id}.tiff"
    if not slide_path.exists():
        return {
            "image_id": image_id,
            "slide_path": "",
            "pen_mark_fraction": None,
            "pen_green_fraction": None,
            "pen_blue_fraction": None,
            "pen_black_fraction": None,
            "pen_colors": "",
            "has_pen_mark": False,
            "status": "missing_slide",
        }
    try:
        fracs = slide_pen_mark_fractions(slide_path, max_edge=thumbnail_edge)
        flagged = pen_mark_is_flagged(fracs)
        return {
            "image_id": image_id,
            "slide_path": str(slide_path),
            "pen_mark_fraction": round(fracs["total"], 8),
            "pen_green_fraction": round(fracs["green"], 8),
            "pen_blue_fraction": round(fracs["blue"], 8),
            "pen_black_fraction": round(fracs["black"], 8),
            "pen_colors": pen_colors_detected(fracs),
            "has_pen_mark": flagged,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "image_id": image_id,
            "slide_path": str(slide_path),
            "pen_mark_fraction": None,
            "pen_green_fraction": None,
            "pen_blue_fraction": None,
            "pen_black_fraction": None,
            "pen_colors": "",
            "has_pen_mark": False,
            "status": f"error:{exc}",
        }


def write_outputs(df: pd.DataFrame, threshold: float) -> None:
    flagged_ids = sorted(df.loc[df["has_pen_mark"], "image_id"].tolist())
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PEN_SLIDES_CSV, index=False)
    PEN_SLIDE_IDS_TXT.write_text("\n".join(flagged_ids) + ("\n" if flagged_ids else ""), encoding="utf-8")
    PEN_SLIDE_IDS_JSON.write_text(json.dumps(flagged_ids, indent=2), encoding="utf-8")

    print()
    print("=== Pen mark scan ===")
    print(f"  Heuristic:          {PEN_MARK_HEURISTIC_VERSION} (green, blue, black)")
    print(f"  Slides scanned:     {len(df)}")
    print(f"  With pen marks:     {len(flagged_ids)}")
    if len(flagged_ids):
        ok = df[df["status"] == "ok"]
        flagged = ok[ok["has_pen_mark"]]
        print(f"    green+:           {(flagged['pen_colors'].str.contains('green')).sum()}")
        print(f"    blue+:            {(flagged['pen_colors'].str.contains('blue')).sum()}")
        print(f"    black+:           {(flagged['pen_colors'].str.contains('black')).sum()}")
    print(f"  Missing slides:     {(df['status'] == 'missing_slide').sum()}")
    print(f"  Threshold:          {threshold}")
    print(f"  Saved CSV:          {PEN_SLIDES_CSV}")
    print(f"  Saved ID list:      {PEN_SLIDE_IDS_TXT}")
    if flagged_ids:
        print("\n  Flagged slide IDs:")
        for image_id in flagged_ids[:20]:
            row = df.loc[df["image_id"] == image_id].iloc[0]
            print(
                f"    {image_id}  total={row.pen_mark_fraction}  "
                f"colors={row.pen_colors or '?'}"
            )
        if len(flagged_ids) > 20:
            print(f"    ... and {len(flagged_ids) - 20} more (see {PEN_SLIDE_IDS_TXT})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan slides for pen marks")
    parser.add_argument(
        "--from-clean",
        action="store_true",
        help="Only scan slides in data/radboud_clean.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/pen_mark_scan_checkpoint.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Flag slide if pen_mark_fraction >= threshold",
    )
    parser.add_argument("--thumbnail-edge", type=int, default=2048)
    args = parser.parse_args()

    ids = load_slide_ids(from_clean=args.from_clean)
    done: dict[str, dict] = {}
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            if checkpoint.get("heuristic_version") != PEN_MARK_HEURISTIC_VERSION:
                raise SystemExit(
                    f"Checkpoint heuristic {checkpoint.get('heuristic_version')!r} "
                    f"!= current {PEN_MARK_HEURISTIC_VERSION!r}; rerun without --resume."
                )
            if checkpoint.get("from_clean") != args.from_clean:
                raise SystemExit("Checkpoint from_clean flag does not match current run.")
            if checkpoint.get("threshold") != args.threshold:
                raise SystemExit("Checkpoint threshold does not match current run.")
            for row in checkpoint.get("rows", []):
                done[row["image_id"]] = row
            print(f"Resuming: {len(done)} slides already in checkpoint")

    scanned = len(done)
    for image_id in tqdm(ids, desc="Scanning pen marks", initial=scanned, total=len(ids)):
        if image_id in done:
            continue
        row = scan_one(image_id, threshold=args.threshold, thumbnail_edge=args.thumbnail_edge)
        done[image_id] = row
        scanned += 1
        if scanned % 25 == 0 or scanned == len(ids):
            save_checkpoint(
                [done[sid] for sid in ids if sid in done],
                from_clean=args.from_clean,
                threshold=args.threshold,
                thumbnail_edge=args.thumbnail_edge,
            )

    rows = [done[image_id] for image_id in ids]
    df = pd.DataFrame(rows)
    write_outputs(df, args.threshold)
    if PEN_CHECKPOINT_JSON.exists():
        PEN_CHECKPOINT_JSON.unlink()


if __name__ == "__main__":
    main()
