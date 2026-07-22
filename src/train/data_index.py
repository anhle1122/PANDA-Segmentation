"""Patch index resolution for baseline training modes."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from patch_utils import PROJECT
from slide_list import load_clean_slide_ids

DATA = PROJECT / "data"
DEFAULT_RULES_TAG = "v33"
FILTER_DIR = PROJECT / "outputs" / f"pen_filter_{DEFAULT_RULES_TAG}"

RAW_H5_DIR = PROJECT / "outputs" / "kept_extract" / "raw"
RAW_H5_STEM = "kept_raw"
VAHADANE_H5_DIR = PROJECT / "outputs" / "stain_norm_kept" / "vahadane"
VAHADANE_H5_STEM = "kept_vahadane"

TRAIN_MODES = {
    "raw": {
        "h5_dir": RAW_H5_DIR,
        "h5_stem": RAW_H5_STEM,
        "raw_h5_dir": RAW_H5_DIR,
        "raw_h5_stem": RAW_H5_STEM,
        "zero_artifact_loss": False,
        "patch_index": DATA / "patch_index_v33_raw.csv",
    },
    "normalized": {
        "h5_dir": VAHADANE_H5_DIR,
        "h5_stem": VAHADANE_H5_STEM,
        "raw_h5_dir": RAW_H5_DIR,
        "raw_h5_stem": RAW_H5_STEM,
        "zero_artifact_loss": True,
        "patch_index": DATA / "patch_index_v33_vahadane.csv",
    },
    # Legacy alias for Model B
    "normalized_ink_raw": {
        "h5_dir": VAHADANE_H5_DIR,
        "h5_stem": VAHADANE_H5_STEM,
        "raw_h5_dir": RAW_H5_DIR,
        "raw_h5_stem": RAW_H5_STEM,
        "zero_artifact_loss": True,
        "patch_index": DATA / "patch_index_v33_vahadane.csv",
    },
}


def h5_path_for(slide_id: str, h5_dir: Path, h5_stem: str) -> Path:
    return h5_dir / f"{slide_id}_{h5_stem}.h5"


def rows_from_h5(slide_id: str, h5_path: Path) -> list[dict]:
    rows = []
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        dom = f["dominant_class"][:]
        for i in range(len(coords)):
            x, y = int(coords[i, 0]), int(coords[i, 1])
            rows.append({
                "image_id": slide_id,
                "x": x,
                "y": y,
                "dominant_class": int(dom[i]),
                "h5_path": str(h5_path),
                "h5_index": i,
            })
    return rows


def build_patch_index(
    mode: str,
    *,
    rules_tag: str = DEFAULT_RULES_TAG,
    filter_dir: Path | None = None,
    require_h5: bool = True,
) -> pd.DataFrame:
    if mode not in TRAIN_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(TRAIN_MODES)}")
    cfg = TRAIN_MODES[mode]
    if filter_dir is None:
        filter_dir = FILTER_DIR

    rows: list[dict] = []
    for sid in tqdm(load_clean_slide_ids(), desc=f"index:{mode}"):
        csv_path = filter_dir / f"pen_filter_{rules_tag}_{sid[:12]}.csv"
        if not csv_path.exists():
            continue
        h5_path = h5_path_for(sid, cfg["h5_dir"], cfg["h5_stem"])
        if require_h5 and not h5_path.exists():
            continue
        if h5_path.exists():
            rows.extend(rows_from_h5(sid, h5_path))
        else:
            df = pd.read_csv(csv_path)
            kept = df[df[f"{rules_tag}_action"] == "keep"]
            for r in kept.itertuples():
                rows.append({
                    "image_id": sid,
                    "x": int(r.x),
                    "y": int(r.y),
                    "dominant_class": -1,
                    "h5_path": "",
                    "h5_index": -1,
                })
    return pd.DataFrame(rows)


def resolve_patch_index(
    mode: str,
    *,
    patch_index_csv: Path | None = None,
    rebuild_index: bool = False,
    require_h5: bool = True,
) -> pd.DataFrame:
    cfg = TRAIN_MODES[mode]
    index_path = patch_index_csv or cfg["patch_index"]
    if rebuild_index or not index_path.exists():
        df = build_patch_index(mode, require_h5=require_h5)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(index_path, index=False)
        return df
    return pd.read_csv(index_path)
