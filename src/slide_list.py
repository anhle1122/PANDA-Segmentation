"""Slide list helpers for Slurm array jobs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from patch_utils import PROJECT

RADBOUD_CLEAN_CSV = PROJECT / "data" / "radboud_clean.csv"


def load_clean_slide_ids() -> list[str]:
    if not RADBOUD_CLEAN_CSV.exists():
        raise FileNotFoundError(f"Missing {RADBOUD_CLEAN_CSV}")
    df = pd.read_csv(RADBOUD_CLEAN_CSV)
    return sorted(df["image_id"].astype(str).unique())


def slide_id_for_task(task_id: int, *, one_based: bool = True) -> str:
    """Map Slurm array task id to slide image_id."""
    slides = load_clean_slide_ids()
    idx = task_id - 1 if one_based else task_id
    if idx < 0 or idx >= len(slides):
        raise IndexError(f"task_id {task_id} out of range (n_slides={len(slides)})")
    return slides[idx]
