"""Rasterize PANDA+ GeoJSON annotations to L0 masks and a patch index for eval.

EVAL ONLY — do not add panda_plus_patches.csv to any training dataloader.

PANDA+ labels -> PANDA Gleason classes (stroma not annotated):
  Benign -> 2, GP3 -> 3, GP4 -> 4, GP5 -> 5, unannotated -> 0

Patch inclusion: labeled_frac = (pixels with class >= 2) / patch_size^2 >= min_labeled_frac.
Class 0 (unannotated) does NOT count toward labeled_frac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

from patch_utils import PATCH_SIZE, PROJECT, SLIDES_DIR

DEFAULT_GEOJSON_DIR = (
    PROJECT / "outputs/panda_plus_geojson_export-2 (1)/panda_plus_geojson_export"
)
DEFAULT_MANIFEST = DEFAULT_GEOJSON_DIR / "MANIFEST.json"
DEFAULT_OVERLAP = PROJECT / "outputs/panda_plus_overlap_audit.csv"
OUT_ROOT = PROJECT / "outputs/panda_plus"

PANDA_PLUS_LABEL_TO_CLASS: dict[str, int] = {
    "Benign": 2,
    "GP3": 3,
    "GP4": 4,
    "GP5": 5,
}
PAINT_ORDER = ("Benign", "GP3", "GP4", "GP5")
SKIP_INDICES = {"4", "400"}


def load_slide_map(manifest_path: Path, overlap_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlap = pd.read_csv(overlap_path)
    overlap["panda_plus_index"] = overlap["panda_plus_index"].astype(str)
    overlap_by_index = overlap.set_index("panda_plus_index")

    rows: list[dict] = []
    for slide in manifest["slides"]:
        idx = Path(slide["filename"]).stem
        row = {
            "panda_plus_index": idx,
            "panda_plus_uuid": slide["slide_id"],
            "width": int(slide["width"]),
            "height": int(slide["height"]),
            "openslide_quickhash1": slide["openslide_quickhash1"],
        }
        if idx in overlap_by_index.index:
            row["panda_image_id"] = str(overlap_by_index.loc[idx, "panda_image_id"])
        else:
            row["panda_image_id"] = ""
        row["has_panda_tiff"] = bool(row["panda_image_id"]) and idx not in SKIP_INDICES
        rows.append(row)
    return pd.DataFrame(rows).sort_values("panda_plus_index", key=lambda s: s.astype(int))


def verify_l0_dimensions(image_id: str, manifest_w: int, manifest_h: int) -> tuple[bool, str]:
    tiff = SLIDES_DIR / f"{image_id}.tiff"
    if not tiff.exists():
        return False, f"TIFF missing: {tiff}"
    slide = openslide.OpenSlide(str(tiff))
    try:
        l0w, l0h = slide.level_dimensions[0]
        qh = slide.properties.get("openslide.quickhash-1", "")
    finally:
        slide.close()
    if (l0w, l0h) != (manifest_w, manifest_h):
        return False, f"L0 size mismatch: openslide={l0w}x{l0h} manifest={manifest_w}x{manifest_h}"
    return True, qh


def _polygon_parts(geometry: dict) -> list[list]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        return [coords]
    if gtype == "MultiPolygon":
        return list(coords)
    return []


def rasterize_geojson(
    geojson_path: Path,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    by_label: dict[str, list] = {label: [] for label in PAINT_ORDER}
    skipped = 0
    for feature in data.get("features", []):
        label = feature.get("properties", {}).get("label", "")
        if label not in PANDA_PLUS_LABEL_TO_CLASS:
            skipped += 1
            continue
        by_label[label].append(feature)

    mask = np.zeros((height, width), dtype=np.uint8)
    painted = {label: 0 for label in PAINT_ORDER}
    for label in PAINT_ORDER:
        cls = PANDA_PLUS_LABEL_TO_CLASS[label]
        for feature in by_label[label]:
            for poly in _polygon_parts(feature.get("geometry", {})):
                if not poly or not poly[0]:
                    continue
                exterior = np.round(np.asarray(poly[0], dtype=np.float64)).astype(np.int32)
                if exterior.shape[0] < 3:
                    continue
                cv2.fillPoly(mask, [exterior], int(cls))
                painted[label] += 1
                for hole in poly[1:]:
                    hole_pts = np.round(np.asarray(hole, dtype=np.float64)).astype(np.int32)
                    if hole_pts.shape[0] >= 3:
                        cv2.fillPoly(mask, [hole_pts], 0)

    n_labeled = int((mask >= 2).sum())
    total = width * height
    stats: dict[str, float | int] = {f"n_{label}": painted[label] for label in PAINT_ORDER}
    stats["n_skipped_features"] = skipped
    stats["n_labeled_pixels"] = n_labeled
    stats["n_unannotated_pixels"] = int((mask == 0).sum())
    stats["labeled_frac_slide"] = float(n_labeled / max(total, 1))
    return mask, stats


def build_patch_index(
    mask: np.ndarray,
    image_id: str,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    min_labeled_frac: float = 0.01,
) -> pd.DataFrame:
    """labeled_frac = (# pixels with class >= 2) / patch_size^2 (NOT # non-zero / total)."""
    stride = stride or patch_size
    patch_area = patch_size * patch_size
    h, w = mask.shape
    rows: list[dict] = []
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            patch = mask[y : y + patch_size, x : x + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                continue
            n_labeled = int((patch >= 2).sum())
            labeled_frac = n_labeled / patch_area
            if labeled_frac < min_labeled_frac:
                continue
            rows.append({
                "image_id": image_id,
                "x": int(x),
                "y": int(y),
                "labeled_frac": round(labeled_frac, 6),
                "n_labeled": n_labeled,
                "n_unannotated": int((patch == 0).sum()),
                "n_benign": int((patch == 2).sum()),
                "n_g3": int((patch == 3).sum()),
                "n_g4": int((patch == 4).sum()),
                "n_g5": int((patch == 5).sum()),
            })
    return pd.DataFrame(rows)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def process_all(args: argparse.Namespace) -> None:
    slide_map = load_slide_map(args.manifest, args.overlap_csv)
    mask_dir = args.out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    patch_frames: list[pd.DataFrame] = []
    geojson_files = {p.stem: p for p in sorted(args.geojson_dir.glob("*.geojson"))}

    candidates = slide_map[slide_map["has_panda_tiff"]].copy()
    if args.limit:
        candidates = candidates.head(args.limit)

    for row in tqdm(candidates.itertuples(index=False), total=len(candidates), desc="PANDA+ slides"):
        idx = str(row.panda_plus_index)
        image_id = str(row.panda_image_id)
        geo_path = geojson_files.get(idx)
        if geo_path is None:
            summary_rows.append({
                "panda_plus_index": idx,
                "panda_image_id": image_id,
                "status": "missing_geojson",
            })
            continue

        ok_dim, dim_msg = verify_l0_dimensions(image_id, int(row.width), int(row.height))
        if not ok_dim:
            summary_rows.append({
                "panda_plus_index": idx,
                "panda_image_id": image_id,
                "status": "dimension_mismatch",
                "error": dim_msg,
            })
            continue

        mask_path = mask_dir / f"{image_id}_pandaplus_mask.png"
        if mask_path.exists() and not args.force:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            stats = {
                "n_labeled_pixels": int((mask >= 2).sum()),
                "n_unannotated_pixels": int((mask == 0).sum()),
                "labeled_frac_slide": float((mask >= 2).sum() / max(mask.size, 1)),
            }
            status = "cached_mask"
        else:
            mask, stats = rasterize_geojson(
                geo_path,
                width=int(row.width),
                height=int(row.height),
            )
            save_mask_png(mask, mask_path)
            status = "rasterized"

        patches = build_patch_index(
            mask,
            image_id,
            patch_size=args.patch_size,
            stride=args.stride,
            min_labeled_frac=args.min_labeled_frac,
        )
        if len(patches):
            patch_frames.append(patches)

        summary_rows.append({
            "panda_plus_index": idx,
            "panda_plus_uuid": row.panda_plus_uuid,
            "panda_image_id": image_id,
            "width": int(row.width),
            "height": int(row.height),
            "mask_path": str(mask_path.relative_to(PROJECT)),
            "n_patches": len(patches),
            "quickhash_ok": dim_msg,
            "status": status,
            **stats,
        })

    skipped = slide_map[~slide_map["has_panda_tiff"]]
    for row in skipped.itertuples(index=False):
        reason = "skip_index" if str(row.panda_plus_index) in SKIP_INDICES else "no_panda_tiff"
        summary_rows.append({
            "panda_plus_index": str(row.panda_plus_index),
            "panda_image_id": str(row.panda_image_id),
            "status": reason,
            "n_patches": 0,
        })

    summary = pd.DataFrame(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slide_map.to_csv(args.out_dir / "panda_plus_slide_map.csv", index=False)
    summary.to_csv(args.out_dir / "panda_plus_processing_summary.csv", index=False)
    patches_out = args.out_dir / "panda_plus_patches.csv"
    if patch_frames:
        pd.concat(patch_frames, ignore_index=True).to_csv(patches_out, index=False)
    else:
        patches_out.write_text(
            "image_id,x,y,labeled_frac,n_labeled,n_unannotated,n_benign,n_g3,n_g4,n_g5\n",
            encoding="utf-8",
        )

    ok = summary[summary["status"].isin({"rasterized", "cached_mask"})]
    print(f"Processed {len(ok)} slides with PANDA TIFFs")
    print(f"  masks   -> {mask_dir}")
    print(f"  patches -> {patches_out} ({int(ok['n_patches'].sum()) if len(ok) else 0} patches)")
    print(f"  summary -> {args.out_dir / 'panda_plus_processing_summary.csv'}")
    if len(ok):
        zero_patch = ok[ok["n_patches"] == 0]
        if len(zero_patch):
            print(f"  WARNING: {len(zero_patch)} slides with 0 patches (possible raster failure)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize PANDA+ GeoJSON for eval (not training)")
    parser.add_argument("--geojson-dir", type=Path, default=DEFAULT_GEOJSON_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overlap-csv", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_SIZE)
    parser.add_argument(
        "--min-labeled-frac",
        type=float,
        default=0.01,
        help="Keep patch if (class>=2 pixels) / patch_size^2 >= this value",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process first N eval slides (0=all)")
    parser.add_argument("--force", action="store_true", help="Re-rasterize even if mask exists")
    args = parser.parse_args()
    process_all(args)


if __name__ == "__main__":
    main()
