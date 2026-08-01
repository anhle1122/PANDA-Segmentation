"""Mask-derived ISUP vs clinical ISUP (NOT model predictions).

Aggregates G0–G5 pixel counts from **raw mask patches** in the train split
(same patches training sees), runs ``isup_diagnostic.derive_grade`` @ min_area_pct,
and compares to metadata ISUP.

This is intentionally distinct from ``diagnostic_report.csv``, which uses
**model** ``pred_pixels_*`` (teacher A / Model C A → 2018/3746 = 53.9%).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isup_diagnostic import derive_grade, gleason_to_isup  # noqa: E402
from patch_utils import MASKS_DIR, NUM_CLASSES, PATCH_SIZE, PROJECT  # noqa: E402

DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"
DEFAULT_METADATA = PROJECT / "data" / "train.csv"
DEFAULT_OUT = PROJECT / "outputs" / "pseudo_label" / "mask_isup_vs_clinical.csv"


def metadata_isup(gleason: str) -> int:
    g = str(gleason).strip().lower()
    if g in {"negative", "0+0", "0", "nan", ""}:
        return 0
    a, b = g.split("+")
    return gleason_to_isup(int(a), int(b))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    p.add_argument("--min-area-pct", type=float, default=0.05)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit-slides", type=int, default=0, help="0 = all")
    args = p.parse_args()

    split = pd.read_csv(args.split_csv)
    meta = pd.read_csv(args.metadata_csv).set_index("image_id")
    split = split.sort_values(["image_id", "x", "y"]).reset_index(drop=True)

    slide_ids = list(dict.fromkeys(split["image_id"].astype(str)))
    if args.limit_slides and args.limit_slides > 0:
        keep = set(slide_ids[: args.limit_slides])
        split = split[split["image_id"].astype(str).isin(keep)].reset_index(drop=True)
        slide_ids = list(dict.fromkeys(split["image_id"].astype(str)))

    counts: dict[str, np.ndarray] = {
        sid: np.zeros(NUM_CLASSES, dtype=np.int64) for sid in slide_ids
    }
    n_patches: dict[str, int] = defaultdict(int)

    cur_id: str | None = None
    cur_slide = None
    try:
        for row in tqdm(split.itertuples(index=False), total=len(split), desc="mask patches"):
            sid = str(row.image_id)
            x, y = int(row.x), int(row.y)
            if sid != cur_id:
                if cur_slide is not None:
                    cur_slide.close()
                mask_path = args.masks_dir / f"{sid}_mask.tiff"
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                cur_slide = openslide.OpenSlide(str(mask_path))
                cur_id = sid
            assert cur_slide is not None
            raw = np.array(cur_slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))
            # PANDA masks: label in channel 0 of RGBA region
            if raw.ndim == 3:
                mask = raw[:, :, 0]
            else:
                mask = raw
            mask = np.clip(mask, 0, 5).astype(np.int64)
            counts[sid] += np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
            n_patches[sid] += 1
    finally:
        if cur_slide is not None:
            cur_slide.close()

    rows = []
    for sid in slide_ids:
        c = counts[sid]
        gleason, derived_isup = derive_grade(c, min_area_pct=args.min_area_pct)
        meta_g = str(meta.loc[sid, "gleason_score"]) if sid in meta.index else ""
        meta_i = metadata_isup(meta_g) if meta_g else -1
        cancer = int(c[3:6].sum())
        fracs = (
            (c[3:6].astype(np.float64) / cancer).tolist()
            if cancer > 0
            else [0.0, 0.0, 0.0]
        )
        rows.append(
            {
                "slide_id": sid,
                "source": "raw_mask_patches",
                "not_model_predictions": True,
                "metadata_gleason": meta_g,
                "metadata_isup": meta_i,
                "mask_derived_gleason": gleason,
                "mask_derived_isup": int(derived_isup),
                "match": bool(int(derived_isup) == int(meta_i)),
                "n_patches": int(n_patches[sid]),
                "mask_pixels_0": int(c[0]),
                "mask_pixels_1": int(c[1]),
                "mask_pixels_2": int(c[2]),
                "mask_pixels_3": int(c[3]),
                "mask_pixels_4": int(c[4]),
                "mask_pixels_5": int(c[5]),
                "cancer_frac_g3": float(fracs[0]),
                "cancer_frac_g4": float(fracs[1]),
                "cancer_frac_g5": float(fracs[2]),
                "min_area_pct": float(args.min_area_pct),
            }
        )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    n = len(out)
    n_match = int(out["match"].sum())
    rate = float(n_match / n) if n else 0.0
    summary = {
        "comparison": "mask_derived_isup_vs_clinical_isup",
        "input": "raw mask patch pixels (OpenSlide L0 mask tiffs), NOT model preds",
        "derive_grade": "isup_diagnostic.derive_grade",
        "min_area_pct": float(args.min_area_pct),
        "n_slides": n,
        "n_match": n_match,
        "match_rate": rate,
        "out_csv": str(args.out_csv),
        "contrast_model_diagnostic": {
            "file": str(PROJECT / "outputs/pseudo_label/diagnostic_report.csv"),
            "label": "model_pred_isup_vs_clinical (teacher A / Model C A)",
            "n_match": 2018,
            "n_slides": 3746,
            "match_rate": 2018 / 3746,
        },
        "per_metadata_isup_match_rate": {
            str(k): float(v)
            for k, v in out.groupby("metadata_isup")["match"].mean().items()
        },
    }
    summary_path = args.out_csv.with_name(args.out_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== MASK vs CLINICAL ISUP (raw masks) ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
