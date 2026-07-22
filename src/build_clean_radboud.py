"""Download 6th-place reference metadata for post-hoc noise overlap analysis.

This script does NOT filter training data. The noise_ratio_10 flags encode
another team's model opinion (their architecture, tiling, preprocessing) and
are kept strictly as a cross-reference for src/check_noise_overlap.py after
our own model is trained.

Source: analokmaus/kaggle-panda-challenge-public (PANDA 6th place, team BarelyBears)
Discussion: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/discussion/169230
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"

TRAIN_URL = (
    "https://raw.githubusercontent.com/analokmaus/kaggle-panda-challenge-public/master/train.csv"
)


def main() -> None:
    train = pd.read_csv(TRAIN_URL)

    sixth_place_path = DATA / "sixth_place_train.csv"
    train.to_csv(sixth_place_path, index=False)
    print(f"Saved reference train.csv ({len(train)} rows) -> {sixth_place_path}")
    print("  Columns: loss_rank, noise_ratio_5/10/15/20, data_provider, ...")

    flagged = train.loc[train["noise_ratio_10"] == 0, "image_id"].sort_values()
    flagged_path = DATA / "sixth_place_noise_flagged_ids.txt"
    flagged_path.write_text("\n".join(flagged) + "\n", encoding="utf-8")
    print(f"Saved {len(flagged)} reference-flagged IDs -> {flagged_path}")
    print("  (noise_ratio_10 == 0; REFERENCE ONLY — not used to filter training)")

    # Legacy filename kept for backward compatibility with older scripts/docs.
    legacy_path = DATA / "excluded_slide_ids.txt"
    legacy_path.write_text(flagged_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  (also mirrored to {legacy_path.name} for compatibility)")

    by_provider = (
        train.loc[train["noise_ratio_10"] == 0]
        .groupby("data_provider")["image_id"]
        .count()
        .sort_index()
    )
    print("\n6th-place noise_ratio_10 == 0 counts by data_provider:")
    for provider, count in by_provider.items():
        print(f"  {provider}: {count}")

    if (OUTPUTS / "radboud_slides_with_masks.csv").exists():
        radboud_masks = pd.read_csv(OUTPUTS / "radboud_slides_with_masks.csv")
        radboud_masks.to_csv(DATA / "radboud_slides_with_masks.csv", index=False)
        flagged_set = set(flagged)
        radboud_flagged = radboud_masks["image_id"].isin(flagged_set).sum()
        print()
        print(f"Radboud slides with masks: {len(radboud_masks)}")
        print(f"  overlap with 6th-place flags: {radboud_flagged} (reference only)")

    print()
    print("Training data is filtered by src/clean_dataset.py (QC checks only).")
    print("After training, run: python src/check_noise_overlap.py --top-pct 10")


if __name__ == "__main__":
    main()
