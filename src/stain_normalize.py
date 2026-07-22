"""Stain normalization: reference fit, slide-level source stains, tissue-only transform."""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import DictionaryLearning

from patch_utils import PATCH_INDEX_CSV, PATCH_SIZE, SLIDES_DIR, OUTPUTS, read_rgb_patch

# staintools / literature defaults (see staintools stain_extraction/*.py)
LUMINOSITY_THRESHOLD = 0.8          # tissue mask for stain matrix estimation
MACENKO_ANGULAR_PERCENTILE = 99     # Macenko et al. 2009 via staintools
VAHADANE_REGULARIZER = 0.1          # Vahadane et al. 2016 via staintools trainDL lambda1
CONCENTRATION_PERCENTILE = 99       # StainNormalizer.fit/transform maxC percentile
MIN_TISSUE_FRACTION = 0.50          # patch QC: skip transform if tissue fraction below this
TISSUE_THRESHOLD = MIN_TISSUE_FRACTION  # alias for CLI
PEN_MARK_MIN_R = 150
PEN_MARK_MIN_G = 100
PEN_MARK_MAX_B = 80
PEN_MARK_FLAG_FRACTION = 0.0001
SUPPORTED_METHODS = ("macenko", "vahadane")


def _install_spams_stub() -> None:
    """Fallback when spams is unavailable (non-Windows or missing wheel)."""

    def trainDL(  # noqa: N802
        X,
        K=2,
        lambda1=0.1,
        mode=2,
        modeD=0,
        posAlpha=True,
        posD=True,
        verbose=False,
    ):
        del mode, modeD, posAlpha, posD, verbose
        samples = np.asarray(X, dtype=np.float64).T
        if len(samples) > 8000:
            idx = np.random.choice(len(samples), 8000, replace=False)
            samples = samples[idx]
        learner = DictionaryLearning(
            n_components=K,
            alpha=lambda1,
            positive_dict=True,
            transform_max_iter=400,
            max_iter=400,
            fit_algorithm="lars",
            random_state=0,
        )
        learner.fit(samples)
        return learner.components_.T

    def lasso(X, D, mode=2, lambda1=0.01, pos=True):  # noqa: N802
        del mode, lambda1, pos
        d = np.asarray(D, dtype=np.float64)
        x = np.asarray(X, dtype=np.float64)
        coef, _, _, _ = np.linalg.lstsq(d, x, rcond=None)
        coef = np.clip(coef, 0, None)
        return sp.csr_matrix(coef)

    import types

    spams_mod = types.ModuleType("spams")
    spams_mod.trainDL = trainDL
    spams_mod.lasso = lasso
    sys.modules["spams"] = spams_mod


def get_normalizer(method: str) -> tuple[object, str]:
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")

    backend = "none"
    if method == "vahadane":
        backend = "spams"
        try:
            import spams  # noqa: F401
        except ImportError:
            _install_spams_stub()
            backend = "sklearn_stub"

    from staintools import StainNormalizer

    return StainNormalizer(method=method), backend


def tissue_content_fraction(img: np.ndarray) -> float:
    """Tissue fraction using staintools luminosity mask (LAB L < 0.8)."""
    import cv2

    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0] / 255.0
    return float((l_channel < LUMINOSITY_THRESHOLD).mean())


def pen_mark_fraction(img: np.ndarray) -> float:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    pen = (r > PEN_MARK_MIN_R) & (g > PEN_MARK_MIN_G) & (b < PEN_MARK_MAX_B)
    return float(pen.mean())


def patch_key(image_id: str, x: int, y: int) -> tuple[str, int, int]:
    return (image_id, int(x), int(y))


def parse_reference(value: str) -> tuple[str, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("reference must be image_id,x,y")
    return parts[0].strip(), int(parts[1]), int(parts[2])


def pick_example_patches(df: pd.DataFrame, per_slide: int = 3) -> pd.DataFrame:
    picks = []
    for image_id, grp in df.groupby("image_id"):
        grp = grp.sort_values(["dominant_class", "x", "y"])
        diverse = grp.drop_duplicates("dominant_class", keep="first")
        chosen = pd.concat([diverse, grp]).drop_duplicates(subset=["x", "y"]).head(per_slide)
        picks.append(chosen)
    return pd.concat(picks, ignore_index=True)


def load_reference_patch(ref: tuple[str, int, int]) -> np.ndarray:
    image_id, x, y = ref
    slide_path = SLIDES_DIR / f"{image_id}.tiff"
    img = read_rgb_patch(slide_path, x, y, size=PATCH_SIZE)
    if img.shape[0] != PATCH_SIZE or img.shape[1] != PATCH_SIZE:
        raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE} reference patch, got {img.shape[:2]}")
    return img


MIN_SOURCE_TISSUE_PIXELS = 1000


class StainNormInsufficientTissueError(RuntimeError):
    """Slide has too much artifact for reliable per-slide source stain estimation."""


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=px)


def stain_artifact_mask(
    img: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
) -> np.ndarray:
    """Combined ink/glass mask — True where pixels are excluded from stain estimation."""
    from patch_filter_rules import artifact_pixel_mask, glass_pixel_mask

    img = np.asarray(img, dtype=np.uint8)
    glass = glass_pixel_mask(img) if mask_glass else np.zeros(img.shape[:2], dtype=bool)
    art = artifact_pixel_mask(img) if mask_artifacts else np.zeros(img.shape[:2], dtype=bool)
    if artifact_dilate_px > 0:
        art = _dilate_mask(art, artifact_dilate_px)
    return glass | art


def detect_artifacts(
    img: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
) -> np.ndarray:
    """Alias for stain_artifact_mask — True = ink / glass / artifact pixel."""
    return stain_artifact_mask(
        img,
        mask_artifacts=mask_artifacts,
        mask_glass=mask_glass,
        artifact_dilate_px=artifact_dilate_px,
    )


def remove_artifacts_from_patch(
    img: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace artifact pixels with white; return (cleaned, artifact_mask)."""
    img = np.asarray(img, dtype=np.uint8)
    artifact_mask = stain_artifact_mask(
        img,
        mask_artifacts=mask_artifacts,
        mask_glass=mask_glass,
        artifact_dilate_px=artifact_dilate_px,
    )
    cleaned = img.copy()
    if artifact_mask.any():
        cleaned[artifact_mask] = 255
    return cleaned, artifact_mask


def get_concentrations(I, stain_matrix, regularizer=0.01):
    from staintools.miscellaneous.get_concentrations import get_concentrations as _gc

    return _gc(I, stain_matrix, regularizer=regularizer)


def estimate_stain_stats(
    normalizer,
    img: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
    min_tissue_pixels: int = MIN_SOURCE_TISSUE_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide-level source stain matrix + max concentrations from tissue pixels only.

    1. detect_artifacts(thumbnail) → artifact mask
    2. tissue_pixels = img[~artifact_mask]
    3. if len(tissue_pixels) < min_tissue_pixels → raise (skip slide)
    4. fit source stain from tissue_pixels.reshape(1, N, 3) only
    """
    img = np.asarray(img, dtype=np.uint8)
    artifact_mask = detect_artifacts(
        img,
        mask_artifacts=mask_artifacts,
        mask_glass=mask_glass,
        artifact_dilate_px=artifact_dilate_px,
    )
    tissue_mask = ~artifact_mask
    n_tissue = int(tissue_mask.sum())
    if n_tissue < min_tissue_pixels:
        raise StainNormInsufficientTissueError(
            f"too much artifact for reliable normalization "
            f"({n_tissue} tissue pixels < {min_tissue_pixels})"
        )

    tissue_pixels = img[tissue_mask]
    fake_img = tissue_pixels.reshape(1, -1, 3)
    stain_matrix = normalizer.extractor.get_stain_matrix(fake_img)
    concentrations = get_concentrations(fake_img, stain_matrix)
    max_c = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
    return stain_matrix, max_c


def transform_tissue_only(
    normalizer,
    raw_patch: np.ndarray,
    stain_matrix_source: np.ndarray,
    max_c_source: np.ndarray,
    artifact_mask: np.ndarray,
) -> np.ndarray:
    """Normalize only tissue pixels; artifact pixels stay as raw scanner RGB."""
    raw_patch = np.asarray(raw_patch, dtype=np.uint8)
    out = raw_patch.copy()
    tissue_mask = ~artifact_mask

    if not tissue_mask.any():
        return out

    tissue_pixels = raw_patch[tissue_mask]
    fake_img = tissue_pixels.reshape(1, -1, 3).astype(np.uint8)

    concentrations = get_concentrations(fake_img, stain_matrix_source)
    concentrations *= normalizer.maxC_target / max_c_source
    tmp = 255 * np.exp(-1 * np.dot(concentrations, normalizer.stain_matrix_target))
    normalized_pixels = tmp.reshape(-1, 3).astype(np.uint8)
    out[tissue_mask] = normalized_pixels
    return out


def transform_with_fixed_stain(
    normalizer,
    img: np.ndarray,
    stain_matrix_source: np.ndarray,
    max_c_source: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
) -> np.ndarray:
    """Tissue-only Vahadane: normalize real tissue; ink/glass stay raw."""
    img = np.asarray(img, dtype=np.uint8)
    artifact_mask = stain_artifact_mask(
        img,
        mask_artifacts=mask_artifacts,
        mask_glass=mask_glass,
        artifact_dilate_px=artifact_dilate_px,
    )
    return transform_tissue_only(
        normalizer, img, stain_matrix_source, max_c_source, artifact_mask,
    )


def transform_with_fixed_stain_white_then_norm(
    normalizer,
    img: np.ndarray,
    stain_matrix_source: np.ndarray,
    max_c_source: np.ndarray,
    *,
    mask_artifacts: bool = True,
    mask_glass: bool = True,
    artifact_dilate_px: int = 0,
) -> np.ndarray:
    """Previous Model B: white-out artifacts, normalize full patch, keep blanks white."""
    cleaned, artifact_mask = remove_artifacts_from_patch(
        img,
        mask_artifacts=mask_artifacts,
        mask_glass=mask_glass,
        artifact_dilate_px=artifact_dilate_px,
    )
    concentrations = get_concentrations(cleaned, stain_matrix_source)
    concentrations *= normalizer.maxC_target / max_c_source
    tmp = 255 * np.exp(-1 * np.dot(concentrations, normalizer.stain_matrix_target))
    out = tmp.reshape(cleaned.shape).astype(np.uint8)
    if artifact_mask.any():
        out[artifact_mask] = 255
    return out


def choose_reference(
  patch_info: list[dict],
  tissue_threshold: float,
  explicit: tuple[str, int, int] | None,
) -> tuple[tuple[str, int, int], np.ndarray, float]:
    if explicit is not None:
        ref_img = load_reference_patch(explicit)
        tissue_pct = tissue_content_fraction(ref_img)
        if tissue_pct <= tissue_threshold:
            raise ValueError(
                f"Reference patch {explicit} has only {tissue_pct:.1%} tissue "
                f"(need >{tissue_threshold:.0%})"
            )
        return explicit, ref_img, tissue_pct

    tissue_rich = [p for p in patch_info if p["tissue_pct"] > tissue_threshold]
    if not tissue_rich:
        raise RuntimeError("No patches with sufficient tissue content for reference fit.")

    best = max(tissue_rich, key=lambda p: (p["row"]["dominant_class"], p["tissue_pct"]))
    row = best["row"]
    ref = patch_key(row["image_id"], row["x"], row["y"])
    return ref, best["before"], best["tissue_pct"]


def save_before_after(
    before: np.ndarray,
    after: np.ndarray,
    out_path,
    *,
    after_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(before)
    axes[0].set_title("Before")
    axes[0].axis("off")
    axes[1].imshow(after)
    axes[1].set_title(after_label)
    axes[1].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stain normalization: fit one reference, transform tissue-rich patches"
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="macenko")
    parser.add_argument("--patches-per-slide", type=int, default=3)
    parser.add_argument("--tissue-threshold", type=float, default=TISSUE_THRESHOLD)
    parser.add_argument(
        "--reference",
        type=parse_reference,
        default=None,
        help="Fixed reference patch as image_id,x,y (fit once, transform all others)",
    )
    parser.add_argument(
        "--tissue-only-demo",
        action="store_true",
        help="Only save demo PNGs for tissue-rich patches (skip low-tissue panels)",
    )
    parser.add_argument(
        "--include-reference-in-demo",
        action="store_true",
        help="Also save before/after for the reference patch (usually looks unchanged)",
    )
    args = parser.parse_args()

    if not PATCH_INDEX_CSV.exists():
        raise FileNotFoundError(f"Run extract_patches.py first. Missing {PATCH_INDEX_CSV}")

    df = pd.read_csv(PATCH_INDEX_CSV)
    examples = pick_example_patches(df, per_slide=args.patches_per_slide)

    method = args.method.lower()
    out_dir = OUTPUTS / "stain_norm_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    patch_info: list[dict] = []
    for _, row in examples.iterrows():
        slide_path = SLIDES_DIR / f"{row['image_id']}.tiff"
        before = read_rgb_patch(slide_path, int(row["x"]), int(row["y"]), size=PATCH_SIZE)
        tissue_pct = tissue_content_fraction(before)
        pen_pct = pen_mark_fraction(before)
        patch_info.append(
            {
                "row": row,
                "before": before,
                "tissue_pct": tissue_pct,
                "pen_pct": pen_pct,
            }
        )

    ref_key, ref_img, ref_tissue = choose_reference(
        patch_info, args.tissue_threshold, args.reference
    )
    normalizer, backend = get_normalizer(method)
    print(f"Method: {method}  |  Backend: {backend}  |  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(
        f"Reference (fit target): {ref_key[0][:8]} x{ref_key[1]} y{ref_key[2]} "
        f"(tissue={ref_tissue:.1%})"
    )
    print("Fitting normalizer on reference patch...")
    normalizer.fit(ref_img)

    saved = 0
    for info in patch_info:
        row = info["row"]
        key = patch_key(row["image_id"], row["x"], row["y"])
        before = info["before"]
        tissue_pct = info["tissue_pct"]
        pen_pct = info["pen_pct"]
        notes: list[str] = []

        if not args.include_reference_in_demo and key == ref_key:
            notes.append("reference_patch")
            continue

        if args.tissue_only_demo and tissue_pct <= args.tissue_threshold:
            continue

        if pen_pct >= PEN_MARK_FLAG_FRACTION:
            notes.append("pen_marks_detected")

        if tissue_pct <= args.tissue_threshold:
            after = before.copy()
            after_label = f"After (original, tissue={tissue_pct:.0%})"
            normalized = False
            notes.append("skipped_low_tissue")
            print(f"  skip norm {row['image_id'][:8]} x{row['x']} y{row['y']} (tissue={tissue_pct:.1%})")
        else:
            try:
                after = normalizer.transform(before)
                backend_note = f", {backend}" if method == "vahadane" and backend != "none" else ""
                after_label = f"After ({method.title()}{backend_note})"
                normalized = True
            except Exception as exc:
                after = before.copy()
                after_label = "After (original, error)"
                normalized = False
                notes.append(f"norm_error:{exc}")
                print(f"  norm failed {row['image_id'][:8]} ({exc})")

        fname = f"{row['image_id'][:8]}_x{row['x']}_y{row['y']}_{method}.png"
        save_before_after(before, after, out_dir / fname, after_label=after_label)
        saved += 1
        print(f"  saved {fname}" + (" [pen marks]" if "pen_marks_detected" in notes else ""))

        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "dominant_class": int(row["dominant_class"]),
                "method": method,
                "patch_size": PATCH_SIZE,
                "stain_backend": backend,
                "reference_image_id": ref_key[0],
                "reference_x": ref_key[1],
                "reference_y": ref_key[2],
                "tissue_pct": round(tissue_pct * 100, 2),
                "pen_mark_pct": round(pen_pct * 100, 4),
                "normalized": normalized,
                "notes": "; ".join(notes) if notes else "",
                "output_file": fname,
            }
        )

    manifest_path = OUTPUTS / f"stain_norm_manifest_{method}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved {saved} before/after images -> {out_dir}")
    print(f"Saved manifest        -> {manifest_path}")
    n_norm = sum(1 for r in manifest_rows if r["normalized"])
    n_pen = sum(1 for r in manifest_rows if "pen_marks_detected" in r["notes"])
    print(f"Normalized: {n_norm}/{saved}  |  Pen-mark flagged: {n_pen}")


if __name__ == "__main__":
    main()
