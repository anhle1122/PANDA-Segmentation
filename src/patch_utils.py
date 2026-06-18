"""Shared helpers for patch extraction and preprocessing demos."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import openslide

PROJECT = Path(__file__).resolve().parent.parent
# Local: project/data. HPC: export PANDA_DATA_ROOT=/common/omarmlab/lea14/panda_data
DATA = Path(os.environ.get("PANDA_DATA_ROOT", PROJECT / "data"))
SLIDES_DIR = DATA / "slides"
MASKS_DIR = DATA / "masks"
OUTPUTS = PROJECT / "outputs"
SELECTED_JSON = OUTPUTS / "selected_slides.json"
PATCH_INDEX_CSV = DATA / "patch_index.csv"

PATCH_SIZE = 512
NUM_CLASSES = 6

CLASS_NAMES = {
    0: "background",
    1: "stroma",
    2: "benign",
    3: "G3",
    4: "G4",
    5: "G5",
}


def load_selected_ids() -> list[str]:
    data = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    return list(data["selected"])


def read_rgb_patch(slide_path: Path, x: int, y: int, size: int = PATCH_SIZE) -> np.ndarray:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        patch = np.array(slide.read_region((x, y), 0, (size, size)))[:, :, :3]
    finally:
        slide.close()
    return patch.astype(np.uint8)


def pixel_counts(mask_patch: np.ndarray) -> list[int]:
    return [int((mask_patch == label).sum()) for label in range(NUM_CLASSES)]


def dominant_class(counts: list[int]) -> int:
    return int(np.argmax(counts))


def counts_to_json(counts: list[int]) -> str:
    return json.dumps({str(i): counts[i] for i in range(NUM_CLASSES)})
