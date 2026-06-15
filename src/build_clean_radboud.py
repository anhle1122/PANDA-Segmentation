"""Build clean Radboud training set using 6th place noisy label exclusion list.

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

    excluded = train.loc[train["noise_ratio_10"] == 0, "image_id"].sort_values()
    excluded_path = DATA / "excluded_slide_ids.txt"
    excluded_path.write_text("\n".join(excluded) + "\n", encoding="utf-8")
    print(f"Saved {len(excluded)} excluded slide IDs to {excluded_path}")
    print("  (noise_ratio_10 == 0 from 6th place solution train.csv)")

    clean_total = (train["noise_ratio_10"] == 1).sum()
    print(f"  Full PANDA clean slides: {clean_total} / {len(train)}")

    radboud_masks = pd.read_csv(OUTPUTS / "radboud_slides_with_masks.csv")
    radboud_masks.to_csv(DATA / "radboud_slides_with_masks.csv", index=False)

    excluded_set = set(excluded)
    radboud_excluded = radboud_masks[radboud_masks["image_id"].isin(excluded_set)]
    radboud_clean = radboud_masks[~radboud_masks["image_id"].isin(excluded_set)].copy()

    meta = pd.read_csv(DATA / "train_radboud.csv")
    radboud_clean = radboud_clean.merge(meta, on="image_id", how="left")

    clean_path = DATA / "radboud_clean.csv"
    radboud_clean.to_csv(clean_path, index=False)

    print()
    print(f"Radboud slides with masks (before): {len(radboud_masks)}")
    print(f"Excluded from Radboud (noisy):    {len(radboud_excluded)}")
    print(f"Radboud clean (after exclusion):    {len(radboud_clean)}")
    print(f"Saved clean list to {clean_path}")
    print()
    print(f"=== {len(radboud_clean)} Radboud slides remain after exclusion ===")


if __name__ == "__main__":
    main()
