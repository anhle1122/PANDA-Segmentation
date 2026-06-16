"""Augmentation preview: HED shift + flip + rotation (Step 4)."""

from __future__ import annotations

import argparse

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from albumentations import DualTransform
from skimage.color import hed2rgb, rgb2hed

from patch_utils import PATCH_INDEX_CSV, SLIDES_DIR, OUTPUTS, read_rgb_patch


class HEDShift(DualTransform):
    """Random perturbation in HED color space (histology-specific augmentation)."""

    def __init__(self, hematoxylin=0.05, eosin=0.05, dab=0.02, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.hematoxylin = hematoxylin
        self.eosin = eosin
        self.dab = dab

    def apply(self, img, **params):
        hed = rgb2hed(img.astype(np.float64) / 255.0)
        hed[..., 0] += np.random.uniform(-self.hematoxylin, self.hematoxylin)
        hed[..., 1] += np.random.uniform(-self.eosin, self.eosin)
        if hed.shape[-1] > 2:
            hed[..., 2] += np.random.uniform(-self.dab, self.dab)
        out = hed2rgb(np.clip(hed, 0, None))
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("hematoxylin", "eosin", "dab")


def pick_example_patches(df: pd.DataFrame, per_slide: int = 3) -> pd.DataFrame:
    picks = []
    for image_id, grp in df.groupby("image_id"):
        grp = grp.sort_values(["dominant_class", "x", "y"])
        diverse = grp.drop_duplicates("dominant_class", keep="first")
        chosen = pd.concat([diverse, grp]).drop_duplicates(subset=["x", "y"]).head(per_slide)
        picks.append(chosen)
    return pd.concat(picks, ignore_index=True)


def build_augmentor() -> A.Compose:
    return A.Compose(
        [
            HEDShift(hematoxylin=0.05, eosin=0.05, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ]
    )


def save_triplet(original: np.ndarray, augmented: np.ndarray, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(augmented)
    axes[1].set_title("HEDShift + flip + rotate")
    axes[1].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentation preview on sample patches")
    parser.add_argument("--patches-per-slide", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not PATCH_INDEX_CSV.exists():
        raise FileNotFoundError(f"Run extract_patches.py first. Missing {PATCH_INDEX_CSV}")

    np.random.seed(args.seed)
    df = pd.read_csv(PATCH_INDEX_CSV)
    examples = pick_example_patches(df, per_slide=args.patches_per_slide)
    augment = build_augmentor()
    out_dir = OUTPUTS / "augmentation_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for _, row in examples.iterrows():
        slide_path = SLIDES_DIR / f"{row['image_id']}.tiff"
        original = read_rgb_patch(slide_path, int(row["x"]), int(row["y"]))
        augmented = augment(image=original)["image"]
        fname = f"{row['image_id'][:8]}_x{row['x']}_y{row['y']}_aug.png"
        save_triplet(original, augmented, out_dir / fname)
        saved += 1
        print(f"  saved {fname}")

    print(f"\nSaved {saved} augmentation examples -> {out_dir}")


if __name__ == "__main__":
    main()
