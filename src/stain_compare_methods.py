"""Side-by-side Macenko vs Vahadane on the same patches as the original demo.

Uses staintools library defaults (Macenko/Vahadane papers via staintools):
  - luminosity_threshold = 0.8
  - macenko angular_percentile = 99
  - vahadane regularizer = 0.1
  - concentration percentile = 99 (inside StainNormalizer.fit/transform)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patch_utils import OUTPUTS, PATCH_SIZE, SLIDES_DIR, read_rgb_patch
from stain_normalize import (
    SUPPORTED_METHODS,
    get_normalizer,
    load_reference_patch,
    patch_key,
    pen_mark_fraction,
)

# staintools defaults (see staintools/stain_extraction/*.py, stain_normalizer.py)
LUMINOSITY_THRESHOLD = 0.8
MACENKO_ANGULAR_PERCENTILE = 99
VAHADANE_REGULARIZER = 0.1
CONCENTRATION_PERCENTILE = 99
MIN_TISSUE_FRACTION = 0.50  # patch-level QC: skip norm if too little tissue for stain fit

# Same reference as original Vahadane demo (highest Gleason + tissue among examples)
DEFAULT_REFERENCE = (
    "85924446350920fb124b657160c966d7",
    1536,
    1024,
)

VAHADANE_MANIFEST = OUTPUTS / "stain_norm_manifest.csv"
COMPARE_DIR = OUTPUTS / "stain_norm_compare"
COMPARE_MANIFEST = OUTPUTS / "stain_norm_compare_manifest.csv"


def tissue_fraction_luminance(img: np.ndarray, threshold: float = LUMINOSITY_THRESHOLD) -> float:
    """Tissue fraction using staintools luminosity mask (LAB L channel)."""
    import cv2

    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0] / 255.0
    return float((l_channel < threshold).mean())


def load_demo_patches(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run the original Vahadane demo first.")
    return pd.read_csv(manifest_path)


def normalize_patch(normalizer, before: np.ndarray, tissue_pct: float) -> tuple[np.ndarray, bool, str]:
    if tissue_pct <= MIN_TISSUE_FRACTION:
        return before.copy(), False, "skipped_low_tissue"
    try:
        return normalizer.transform(before), True, ""
    except Exception as exc:
        return before.copy(), False, f"norm_error:{exc}"


def save_triplet(
    before: np.ndarray,
    vahadane: np.ndarray,
    macenko: np.ndarray,
    out_path: Path,
    *,
    vahadane_applied: bool,
    macenko_applied: bool,
    tissue_pct: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(before)
    axes[0].set_title("Before")
    v_label = "Vahadane" if vahadane_applied else f"Vahadane (skip, tissue={tissue_pct:.0%})"
    m_label = "Macenko" if macenko_applied else f"Macenko (skip, tissue={tissue_pct:.0%})"
    axes[1].imshow(vahadane)
    axes[1].set_title(v_label)
    axes[2].imshow(macenko)
    axes[2].set_title(m_label)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        "Stain norm comparison (staintools defaults, shared reference)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Macenko vs Vahadane on Vahadane demo patches")
    parser.add_argument("--manifest", type=Path, default=VAHADANE_MANIFEST)
    parser.add_argument(
        "--reference",
        nargs=3,
        metavar=("IMAGE_ID", "X", "Y"),
        default=list(DEFAULT_REFERENCE),
        help="Reference patch for fit() on both methods",
    )
    args = parser.parse_args()

    ref_key = (str(args.reference[0]), int(args.reference[1]), int(args.reference[2]))
    ref_img = load_reference_patch(ref_key)

    demo = load_demo_patches(args.manifest)
    vahadane_norm, vahadane_backend = get_normalizer("vahadane")
    macenko_norm, _ = get_normalizer("macenko")

    print("staintools standard parameters:")
    print(f"  luminosity_threshold     = {LUMINOSITY_THRESHOLD}")
    print(f"  macenko angular_pct      = {MACENKO_ANGULAR_PERCENTILE}")
    print(f"  vahadane regularizer     = {VAHADANE_REGULARIZER}")
    print(f"  concentration percentile = {CONCENTRATION_PERCENTILE}")
    print(f"  min tissue fraction      = {MIN_TISSUE_FRACTION}")
    print(f"  patch size               = {PATCH_SIZE}")
    print(f"  Vahadane backend         = {vahadane_backend}")
    print(f"Reference: {ref_key[0][:8]} x{ref_key[1]} y{ref_key[2]}")

    vahadane_norm.fit(ref_img)
    macenko_norm.fit(ref_img)

    rows = []
    for _, item in demo.iterrows():
        image_id = item["image_id"]
        x, y = int(item["x"]), int(item["y"])
        before = read_rgb_patch(SLIDES_DIR / f"{image_id}.tiff", x, y, PATCH_SIZE)
        tissue_pct = tissue_fraction_luminance(before)
        pen_pct = pen_mark_fraction(before)

        v_after, v_ok, v_note = normalize_patch(vahadane_norm, before, tissue_pct)
        m_after, m_ok, m_note = normalize_patch(macenko_norm, before, tissue_pct)

        stem = f"{image_id[:8]}_x{x}_y{y}"
        compare_path = COMPARE_DIR / f"{stem}_compare.png"
        save_triplet(
            before,
            v_after,
            m_after,
            compare_path,
            vahadane_applied=v_ok,
            macenko_applied=m_ok,
            tissue_pct=tissue_pct,
        )
        print(f"  saved {compare_path.name}  tissue={tissue_pct:.1%}  v={v_ok}  m={m_ok}")

        rows.append(
            {
                "image_id": image_id,
                "x": x,
                "y": y,
                "dominant_class": int(item["dominant_class"]),
                "tissue_pct_luminance": round(tissue_pct * 100, 2),
                "pen_mark_pct": round(pen_pct * 100, 4),
                "reference_image_id": ref_key[0],
                "reference_x": ref_key[1],
                "reference_y": ref_key[2],
                "vahadane_normalized": v_ok,
                "macenko_normalized": m_ok,
                "vahadane_notes": v_note,
                "macenko_notes": m_note,
                "compare_file": compare_path.name,
                "luminosity_threshold": LUMINOSITY_THRESHOLD,
                "min_tissue_fraction": MIN_TISSUE_FRACTION,
                "macenko_angular_percentile": MACENKO_ANGULAR_PERCENTILE,
                "vahadane_regularizer": VAHADANE_REGULARIZER,
            }
        )

    pd.DataFrame(rows).to_csv(COMPARE_MANIFEST, index=False)
    n_v = sum(r["vahadane_normalized"] for r in rows)
    n_m = sum(r["macenko_normalized"] for r in rows)
    print(f"\nSaved {len(rows)} comparison images -> {COMPARE_DIR}")
    print(f"Saved manifest             -> {COMPARE_MANIFEST}")
    print(f"Vahadane normalized: {n_v}/{len(rows)}  |  Macenko normalized: {n_m}/{len(rows)}")


if __name__ == "__main__":
    main()
