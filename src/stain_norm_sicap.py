"""SICAPv2 -> PANDA reference: Macenko and Vahadane cross-dataset stain norm demo."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patch_utils import OUTPUTS, PATCH_SIZE, PROJECT
from stain_normalize import (
    LUMINOSITY_THRESHOLD,
    MIN_TISSUE_FRACTION,
    SUPPORTED_METHODS,
    get_normalizer,
    load_reference_patch,
    tissue_content_fraction,
)

PANDA_REFERENCE = (
    "85924446350920fb124b657160c966d7",
    1536,
    1024,
)

SICAP_ROOT = PROJECT / "data" / "sicapv2"
OUT_DIR = OUTPUTS / "stain_norm_sicap"
MANIFEST = OUTPUTS / "stain_norm_sicap_manifest.csv"

IMAGE_DIR_CANDIDATES = (
    SICAP_ROOT / "images",
    SICAP_ROOT / "SICAPv2" / "images",
    SICAP_ROOT / "SICAPv2 Dataset" / "images",
)


def find_sicap_image_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"SICAP image dir not found: {explicit}")
        return explicit
    for candidate in IMAGE_DIR_CANDIDATES:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    if SICAP_ROOT.is_dir():
        for sub in SICAP_ROOT.rglob("images"):
            if sub.is_dir() and any(sub.glob("*.jpg")):
                return sub
    raise FileNotFoundError(
        f"No SICAPv2 images found under {SICAP_ROOT}. "
        "Unzip Mendeley download to data/sicapv2/images/*.jpg"
    )


def read_sicap_patch(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != PATCH_SIZE or rgb.shape[1] != PATCH_SIZE:
        rgb = cv2.resize(rgb, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def pick_sicap_patches(image_dir: Path, n: int, seed: int) -> list[Path]:
    all_jpg = sorted(image_dir.glob("*.jpg"))
    if not all_jpg:
        raise FileNotFoundError(f"No .jpg patches in {image_dir}")

    scored: list[tuple[float, Path]] = []
    for path in all_jpg:
        img = read_sicap_patch(path)
        scored.append((tissue_content_fraction(img), path))

    rich = [p for frac, p in scored if frac > MIN_TISSUE_FRACTION]
    pool = rich if len(rich) >= n else [p for _, p in scored]
    rng = random.Random(seed)
    if len(pool) > n:
        pool = rng.sample(pool, n)
    return sorted(pool[:n])


def transform_patch(normalizers: dict, before: np.ndarray, tissue_pct: float) -> dict[str, tuple[np.ndarray, bool, str]]:
    results: dict[str, tuple[np.ndarray, bool, str]] = {}
    if tissue_pct <= MIN_TISSUE_FRACTION:
        for method in normalizers:
            results[method] = (before.copy(), False, "skipped_low_tissue")
        return results

    for method, (normalizer, backend) in normalizers.items():
        try:
            after = normalizer.transform(before)
            results[method] = (after, True, backend if method == "vahadane" else "")
        except Exception as exc:
            results[method] = (before.copy(), False, f"norm_error:{exc}")
    return results


def save_triplet(
    before: np.ndarray,
    macenko: np.ndarray,
    vahadane: np.ndarray,
    out_path: Path,
    *,
    macenko_ok: bool,
    vahadane_ok: bool,
    tissue_pct: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(before)
    axes[0].set_title("SICAPv2 (before)")
    m_title = "Macenko" if macenko_ok else f"Macenko (skip, {tissue_pct:.0%} tissue)"
    v_title = "Vahadane" if vahadane_ok else f"Vahadane (skip, {tissue_pct:.0%} tissue)"
    axes[1].imshow(macenko)
    axes[1].set_title(m_title)
    axes[2].imshow(vahadane)
    axes[2].set_title(v_title)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Cross-dataset: SICAP -> PANDA reference (Macenko & Vahadane)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overview(
    sicap_before: np.ndarray,
    sicap_macenko: np.ndarray,
    sicap_vahadane: np.ndarray,
    panda_before: np.ndarray,
    panda_macenko: np.ndarray,
    panda_vahadane: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    rows = [
        (sicap_before, sicap_macenko, sicap_vahadane, "SICAPv2"),
        (panda_before, panda_macenko, panda_vahadane, "PANDA (reference slide)"),
    ]
    col_titles = ["Before", "Macenko", "Vahadane"]
    for row_idx, (b, m, v, row_label) in enumerate(rows):
        for col_idx, (img, col_title) in enumerate(zip((b, m, v), col_titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10)
            ax.axis("off")
    fig.suptitle("Cross-site stain norm (shared PANDA reference patch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_methods(value: str) -> list[str]:
    if value == "both":
        return list(SUPPORTED_METHODS)
    method = value.lower()
    if method not in SUPPORTED_METHODS:
        raise argparse.ArgumentTypeError(f"method must be both, macenko, or vahadane")
    return [method]


def main() -> None:
    parser = argparse.ArgumentParser(description="SICAPv2 -> PANDA reference stain norm demo")
    parser.add_argument("--sicap-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        nargs=3,
        metavar=("IMAGE_ID", "X", "Y"),
        default=list(PANDA_REFERENCE),
    )
    parser.add_argument(
        "--method",
        type=parse_methods,
        default=parse_methods("both"),
        help="macenko, vahadane, or both (default: both)",
    )
    args = parser.parse_args()

    ref_key = (str(args.reference[0]), int(args.reference[1]), int(args.reference[2]))
    image_dir = find_sicap_image_dir(args.sicap_dir)
    ref_img = load_reference_patch(ref_key)

    normalizers: dict[str, tuple[object, str]] = {}
    for method in args.method:
        normalizer, backend = get_normalizer(method)
        normalizer.fit(ref_img)
        normalizers[method] = (normalizer, backend)

    print(f"SICAP images: {image_dir}")
    print(f"Reference (PANDA): {ref_key[0][:8]} x{ref_key[1]} y{ref_key[2]}")
    print(f"Methods: {', '.join(args.method)}  |  min tissue: {MIN_TISSUE_FRACTION}")

    panda_results = transform_patch(normalizers, ref_img, tissue_content_fraction(ref_img))
    panda_macenko = panda_results.get("macenko", (ref_img, False, ""))[0]
    panda_vahadane = panda_results.get("vahadane", (ref_img, False, ""))[0]

    paths = pick_sicap_patches(image_dir, args.num_samples, args.seed)
    rows: list[dict] = []
    overview_saved = False

    for path in paths:
        before = read_sicap_patch(path)
        tissue_pct = tissue_content_fraction(before)
        results = transform_patch(normalizers, before, tissue_pct)

        stem = path.stem[:48]
        macenko_img, macenko_ok, macenko_note = results.get("macenko", (before, False, "not_run"))
        vahadane_img, vahadane_ok, vahadane_note = results.get("vahadane", (before, False, "not_run"))

        if len(args.method) == 2:
            out_png = OUT_DIR / f"{stem}_compare.png"
            save_triplet(
                before,
                macenko_img,
                vahadane_img,
                out_png,
                macenko_ok=macenko_ok,
                vahadane_ok=vahadane_ok,
                tissue_pct=tissue_pct,
            )
        else:
            method = args.method[0]
            after, ok, note = results[method]
            out_png = OUT_DIR / f"{stem}_{method}.png"
            title = f"After {method.title()}" if ok else f"After (skip, tissue={tissue_pct:.0%})"
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(before)
            axes[0].set_title("SICAPv2 (before)")
            axes[1].imshow(after)
            axes[1].set_title(title)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            macenko_note = note if method == "macenko" else macenko_note
            vahadane_note = note if method == "vahadane" else vahadane_note

        if not overview_saved and (macenko_ok or vahadane_ok):
            save_overview(
                before,
                macenko_img,
                vahadane_img,
                ref_img,
                panda_macenko,
                panda_vahadane,
                OUT_DIR / "00_overview_macenko_vahadane.png",
            )
            overview_saved = True

        diff_m = float(np.abs(before.astype(float) - macenko_img.astype(float)).mean())
        diff_v = float(np.abs(before.astype(float) - vahadane_img.astype(float)).mean())
        rows.append(
            {
                "sicap_file": path.name,
                "tissue_pct": round(tissue_pct * 100, 2),
                "macenko_normalized": macenko_ok,
                "vahadane_normalized": vahadane_ok,
                "macenko_mean_abs_diff": round(diff_m, 2),
                "vahadane_mean_abs_diff": round(diff_v, 2),
                "reference_image_id": ref_key[0],
                "reference_x": ref_key[1],
                "reference_y": ref_key[2],
                "output_file": out_png.name,
                "macenko_notes": macenko_note,
                "vahadane_notes": vahadane_note,
            }
        )
        print(
            f"  {path.name}  tissue={tissue_pct:.1%}  "
            f"macenko diff={diff_m:.1f} ({macenko_ok})  vahadane diff={diff_v:.1f} ({vahadane_ok})"
        )

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"\nSaved {len(rows)} panels -> {OUT_DIR}")
    print(f"Manifest -> {MANIFEST}")
    if overview_saved:
        print("Overview: outputs/stain_norm_sicap/00_overview_macenko_vahadane.png")


if __name__ == "__main__":
    main()
