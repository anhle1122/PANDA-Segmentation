"""Build Trident custom_list_of_wsis CSV from radboud_clean.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from patch_utils import SLIDES_DIR

PROJECT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT / "data" / "radboud_clean.csv"
DEFAULT_OUT = PROJECT / "data" / "trident_radboud_clean_wsi_list.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    clean = pd.read_csv(CLEAN_CSV)
    rows = []
    missing = []
    for image_id in sorted(clean["image_id"].unique()):
        fname = f"{image_id}.tiff"
        path = SLIDES_DIR / fname
        if path.exists():
            rows.append({"wsi": fname, "image_id": image_id})
        else:
            missing.append(image_id)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df[["wsi"]].to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} slides -> {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} clean slides missing local WSI (skipped)")


if __name__ == "__main__":
    main()
