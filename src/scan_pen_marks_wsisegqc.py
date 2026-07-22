"""Batch pen-mark detection using wsisegqc pen.pt (replaces HSV heuristic).

Usage:
  python src/scan_pen_marks_wsisegqc.py --from-clean
  python src/scan_pen_marks_wsisegqc.py --from-clean --resume
  python src/scan_pen_marks_wsisegqc.py --slide-ids 7d581d0082d6ee0a32165b0f8fe216d8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from patch_utils import MASKS_DIR, OUTPUTS, SLIDES_DIR
from pen_wsisegqc import DEFAULT_PEN_PT, infer_slide_pen

PROJECT = Path(__file__).resolve().parent.parent
RADBOUD_CLEAN_CSV = PROJECT / "data" / "radboud_clean.csv"
OUT_DIR = OUTPUTS / "pen_mark_detection_v2"
OUT_CSV = OUT_DIR / "pen_mark_slides_wsisegqc.csv"
NPZ_DIR = OUT_DIR / "npz"
CHECKPOINT_JSON = OUT_DIR / "pen_wsisegqc_checkpoint.json"

# Default: flag if pen covers >= 0.05% of tissue (pen.pt outputs sparse masks;
# calibrate after validation — much lower than naive 5% which would miss real ink).
DEFAULT_FLAG_PCT = 0.05


def load_slide_ids(*, from_clean: bool, slide_ids: list[str] | None) -> list[str]:
    if slide_ids:
        return sorted(slide_ids)
    if not from_clean:
        raise SystemExit("Provide --from-clean or --slide-ids")
    if not RADBOUD_CLEAN_CSV.exists():
        raise FileNotFoundError(f"Missing {RADBOUD_CLEAN_CSV}")
    df = pd.read_csv(RADBOUD_CLEAN_CSV)
    return sorted(df["image_id"].unique())


def scan_one(
    image_id: str,
    *,
    weights_path: Path,
    device: str,
    flag_pct: float,
    save_npz: bool,
) -> dict:
    slide_path = SLIDES_DIR / f"{image_id}.tiff"
    mask_path = MASKS_DIR / f"{image_id}_mask.tiff"
    if not slide_path.exists():
        return {
            "slide_id": image_id,
            "tissue_area_px": None,
            "pen_area_px": None,
            "pen_pct_of_tissue": None,
            "flagged": False,
            "status": "missing_slide",
        }
    try:
        result = infer_slide_pen(
            slide_path,
            mask_path if mask_path.exists() else None,
            weights_path=weights_path,
            device_str=device,
        )
        if save_npz:
            NPZ_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                NPZ_DIR / f"{image_id}.npz",
                pen=result["pen_mask"],
                tissue=result["tissue_mask"],
            )
        flagged = result["pen_pct_of_tissue"] >= flag_pct
        return {
            "slide_id": image_id,
            "tissue_area_px": result["tissue_area_px"],
            "pen_area_px": result["pen_area_px"],
            "pen_pct_of_tissue": round(result["pen_pct_of_tissue"], 6),
            "flagged": flagged,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "slide_id": image_id,
            "tissue_area_px": None,
            "pen_area_px": None,
            "pen_pct_of_tissue": None,
            "flagged": False,
            "status": f"error:{exc}",
        }


def save_checkpoint(rows: list[dict], meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_JSON.write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def write_outputs(df: pd.DataFrame, flag_pct: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    flagged = df[df["flagged"] & (df["status"] == "ok")]
    flagged_ids = sorted(flagged["slide_id"].tolist())
    (OUT_DIR / "pen_mark_slide_ids.txt").write_text(
        "\n".join(flagged_ids) + ("\n" if flagged_ids else ""),
        encoding="utf-8",
    )
    print()
    print("=== WSISegQC pen mark scan ===")
    print(f"  Slides scanned:     {len(df)}")
    print(f"  Flagged (>={flag_pct}% tissue): {len(flagged)}")
    print(f"  Missing / errors:   {(df['status'] != 'ok').sum()}")
    print(f"  Saved CSV:          {OUT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pen detection via wsisegqc pen.pt")
    parser.add_argument("--from-clean", action="store_true")
    parser.add_argument("--slide-ids", nargs="+", help="Specific slide IDs to scan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flag-pct", type=float, default=DEFAULT_FLAG_PCT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_PEN_PT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-npz", action="store_true", help="Skip saving per-slide .npz")
    args = parser.parse_args()

    ids = load_slide_ids(from_clean=args.from_clean, slide_ids=args.slide_ids)
    done: dict[str, dict] = {}
    if args.resume and CHECKPOINT_JSON.exists():
        ckpt = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        for row in ckpt.get("rows", []):
            done[row["slide_id"]] = row
        print(f"Resuming: {len(done)} slides in checkpoint")

    meta = {
        "flag_pct": args.flag_pct,
        "weights": str(args.weights),
        "device": args.device,
    }
    scanned = len(done)
    for image_id in tqdm(ids, desc="pen.pt scan", initial=scanned, total=len(ids)):
        if image_id in done:
            continue
        row = scan_one(
            image_id,
            weights_path=args.weights,
            device=args.device,
            flag_pct=args.flag_pct,
            save_npz=not args.no_npz,
        )
        done[image_id] = row
        scanned += 1
        if scanned % 10 == 0 or scanned == len(ids):
            save_checkpoint([done[sid] for sid in ids if sid in done], meta)

    rows = [done[sid] for sid in ids]
    df = pd.DataFrame(rows)
    write_outputs(df, args.flag_pct)
    if CHECKPOINT_JSON.exists():
        CHECKPOINT_JSON.unlink()


if __name__ == "__main__":
    main()
