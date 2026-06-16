"""Vahadane stain normalization demo on sample patches (Step 3)."""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import DictionaryLearning

from patch_utils import PATCH_INDEX_CSV, PATCH_SIZE, SLIDES_DIR, OUTPUTS, read_rgb_patch

TISSUE_THRESHOLD = 0.50
WHITE_THRESHOLD = 230
PEN_MARK_MIN_R = 150
PEN_MARK_MIN_G = 100
PEN_MARK_MAX_B = 80
PEN_MARK_FLAG_FRACTION = 0.0001


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


def get_normalizer() -> tuple[object, str]:
    backend = "spams"
    try:
        import spams  # noqa: F401
    except ImportError:
        _install_spams_stub()
        backend = "sklearn_stub"
    from staintools import StainNormalizer

    return StainNormalizer(method="vahadane"), backend


def tissue_content_fraction(img: np.ndarray) -> float:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    tissue = (r < WHITE_THRESHOLD) & (g < WHITE_THRESHOLD) & (b < WHITE_THRESHOLD)
    return float(tissue.mean())


def pen_mark_fraction(img: np.ndarray) -> float:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    pen = (r > PEN_MARK_MIN_R) & (g > PEN_MARK_MIN_G) & (b < PEN_MARK_MAX_B)
    return float(pen.mean())


def pick_example_patches(df: pd.DataFrame, per_slide: int = 3) -> pd.DataFrame:
    picks = []
    for image_id, grp in df.groupby("image_id"):
        grp = grp.sort_values(["dominant_class", "x", "y"])
        diverse = grp.drop_duplicates("dominant_class", keep="first")
        chosen = pd.concat([diverse, grp]).drop_duplicates(subset=["x", "y"]).head(per_slide)
        picks.append(chosen)
    return pd.concat(picks, ignore_index=True)


def save_before_after(
    before: np.ndarray,
    after: np.ndarray,
    out_path,
    *,
    after_label: str = "After (Vahadane)",
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
    parser = argparse.ArgumentParser(description="Vahadane stain normalization examples")
    parser.add_argument("--patches-per-slide", type=int, default=3)
    parser.add_argument("--tissue-threshold", type=float, default=TISSUE_THRESHOLD)
    args = parser.parse_args()

    if not PATCH_INDEX_CSV.exists():
        raise FileNotFoundError(f"Run extract_patches.py first. Missing {PATCH_INDEX_CSV}")

    df = pd.read_csv(PATCH_INDEX_CSV)
    examples = pick_example_patches(df, per_slide=args.patches_per_slide)
    out_dir = OUTPUTS / "stain_norm_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    patch_info: list[dict] = []
    for _, row in examples.iterrows():
        slide_path = SLIDES_DIR / f"{row['image_id']}.tiff"
        before = read_rgb_patch(slide_path, int(row["x"]), int(row["y"]), size=PATCH_SIZE)
        if before.shape[0] != PATCH_SIZE or before.shape[1] != PATCH_SIZE:
            raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE} patch, got {before.shape[:2]}")
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

    tissue_rich = [p for p in patch_info if p["tissue_pct"] > args.tissue_threshold]
    if not tissue_rich:
        raise RuntimeError("No patches with sufficient tissue content for Vahadane reference fit.")

    ref = max(tissue_rich, key=lambda p: (p["row"]["dominant_class"], p["tissue_pct"]))
    normalizer, backend = get_normalizer()
    print(f"Stain backend: {backend}  |  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(
        f"Fitting Vahadane on reference patch {ref['row']['image_id'][:8]} "
        f"(tissue={ref['tissue_pct']:.1%}, class={ref['row']['dominant_class']})..."
    )
    normalizer.fit(ref["before"])

    saved = 0
    for info in patch_info:
        row = info["row"]
        before = info["before"]
        tissue_pct = info["tissue_pct"]
        pen_pct = info["pen_pct"]
        notes: list[str] = []

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
                after_label = "After (Vahadane, spams)" if backend == "spams" else "After (Vahadane)"
                normalized = True
            except Exception as exc:
                after = before.copy()
                after_label = "After (original, error)"
                normalized = False
                notes.append(f"norm_error:{exc}")
                print(f"  norm failed {row['image_id'][:8]} ({exc})")

        fname = f"{row['image_id'][:8]}_x{row['x']}_y{row['y']}_vahadane.png"
        save_before_after(before, after, out_dir / fname, after_label=after_label)
        saved += 1
        print(f"  saved {fname}" + (" [pen marks]" if "pen_marks_detected" in notes else ""))

        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "dominant_class": int(row["dominant_class"]),
                "patch_size": PATCH_SIZE,
                "stain_backend": backend,
                "tissue_pct": round(tissue_pct * 100, 2),
                "pen_mark_pct": round(pen_pct * 100, 4),
                "normalized": normalized,
                "notes": "; ".join(notes) if notes else "",
                "output_file": fname,
            }
        )

    manifest_path = OUTPUTS / "stain_norm_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved {saved} before/after images -> {out_dir}")
    print(f"Saved manifest        -> {manifest_path}")
    n_norm = sum(1 for r in manifest_rows if r["normalized"])
    n_pen = sum(1 for r in manifest_rows if "pen_marks_detected" in r["notes"])
    print(f"Normalized: {n_norm}/{saved}  |  Pen-mark flagged: {n_pen}")


if __name__ == "__main__":
    main()
