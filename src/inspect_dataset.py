"""Explore PANDA train.csv: slide counts, ISUP grades, and mask availability."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_CHART = PROJECT_ROOT / "notebooks" / "isup_distribution.png"


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)

    print("=" * 60)
    print("PANDA train.csv - dataset summary")
    print("=" * 60)
    print(f"Total slides: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    print("Slides by data provider:")
    print(df["data_provider"].value_counts().to_string())
    print()

    print("ISUP grade distribution (all slides):")
    isup_counts = df["isup_grade"].value_counts().sort_index()
    print(isup_counts.to_string())
    print()

    df["has_mask"] = df["data_provider"] == "radboud"
    mask_counts = df["has_mask"].value_counts()
    print("Mask availability (Radboud = has pixel-level mask):")
    print(f"  With masks:    {mask_counts.get(True, 0)}")
    print(f"  Without masks: {mask_counts.get(False, 0)}")
    print()

    radboud = df[df["data_provider"] == "radboud"].copy()
    radboud_path = DATA_DIR / "train_radboud.csv"
    radboud.to_csv(radboud_path, index=False)
    print(f"Radboud-only subset: {len(radboud)} slides (saved to {radboud_path})")
    print(f"  Radboud with masks:    {radboud['has_mask'].sum()}")
    print(f"  Radboud without masks: {(~radboud['has_mask']).sum()}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    isup_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("PANDA Training Set - ISUP Grade Distribution")
    ax.set_xlabel("ISUP Grade")
    ax.set_ylabel("Number of Slides")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=150)
    plt.close(fig)
    print(f"Saved ISUP distribution chart to {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
