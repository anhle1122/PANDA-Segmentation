"""Visual QC: stitch per-tile Macenko-normalized AGGC tiles back into a whole-slide
image, and compare against the raw AGGC slide + the PANDA reference slide.

Per-tile Macenko re-estimates a SOURCE stain matrix from each 512px tile
independently (staintools StainNormalizer.transform), only the TARGET side is
fixed (one PANDA reference patch). That can show up as tile-to-tile color
seams when stitched back -- this script makes that visible before trusting
the full-run numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stain_normalize import get_normalizer, load_reference_patch  # noqa: E402
from stain_norm_sicap import PANDA_REFERENCE  # noqa: E402
from patch_utils import SLIDES_DIR  # noqa: E402
from evaluate_aggc2022 import (  # noqa: E402
    estimate_slide_stain_matrix,
    transform_tile_with_slide_stats,
)

AGGC_ROOT = Path("/common/omarmlab/data/AGGC_2022")


def find_image_path(img_dir: Path, slide_id: str) -> Path:
    for ext in (".tiff", ".tif"):
        p = img_dir / f"{slide_id}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"{slide_id} not found under {img_dir}")


def stitch_normalized(
    img: Image.Image, normalizer, tile: int = 512, level: str = "slide"
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Tile the whole image, Macenko-transform, stitch back.

    level="slide": one stain matrix estimated for the WHOLE slide, applied
    identically to every tile (no per-tile re-estimation -> no seams).
    level="tile": legacy per-tile re-estimation (shows the seam/blowup bug).

    Returns (stitched, worst_tile_topleft) where worst_tile_topleft is the
    (y, x) of the normalized tile with the biggest brightness drop vs raw
    (a cheap proxy for "normalization went wrong on this tile") -- used to
    auto-zoom into a likely seam/artifact instead of guessing coordinates.
    """
    slide_stats = None
    if level == "slide":
        slide_stats = estimate_slide_stain_matrix(normalizer, img)
        print(f"slide_stain_stats={'ok' if slide_stats else 'FAILED (too little tissue)'}")

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    ph = (tile - h % tile) % tile
    pw = (tile - w % tile) % tile
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    hp, wp = arr.shape[:2]
    out = arr.copy()
    n_tiles = 0
    n_ok = 0
    worst_drop = -1.0
    worst_xy: tuple[int, int] | None = None
    for y in range(0, hp, tile):
        for x in range(0, wp, tile):
            t = arr[y : y + tile, x : x + tile]
            n_tiles += 1
            if t.mean() > 220.0 and t.std() < 12.0:
                continue  # background tile, leave as-is (same skip rule as eval)
            try:
                if slide_stats is not None:
                    normed = transform_tile_with_slide_stats(normalizer, t, *slide_stats)
                else:
                    from stain_normalize import tissue_content_fraction

                    if tissue_content_fraction(t) < 0.20:
                        continue
                    normed = normalizer.transform(t)
                out[y : y + tile, x : x + tile] = normed
                n_ok += 1
                drop = float(t.mean()) - float(normed.mean())
                if drop > worst_drop:
                    worst_drop = drop
                    worst_xy = (y, x)
            except Exception:
                pass  # keep raw tile on staintools failure
    print(
        f"stitched tiles={n_tiles} normalized_ok={n_ok} "
        f"worst_brightness_drop={worst_drop:.1f} at {worst_xy}",
        flush=True,
    )
    return out[:h, :w], worst_xy


def thumb(arr: np.ndarray, max_w: int = 900) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max_w / w
    return np.asarray(
        Image.fromarray(arr).resize((max_w, max(1, round(h * scale))), Image.BILINEAR)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="Subset2")
    p.add_argument("--split", default="Test")
    p.add_argument("--slide-id", default="Subset2_Test_1")
    p.add_argument("--method", default="macenko", choices=("macenko", "vahadane"))
    p.add_argument("--level", default="slide", choices=("slide", "tile"))
    p.add_argument("--out", type=Path, default=Path("/tmp/aggc_stain_stitch_compare.png"))
    args = p.parse_args()

    img_dir = AGGC_ROOT / args.subset / args.split / "images"
    img_path = find_image_path(img_dir, args.slide_id)
    print(f"slide={img_path}", flush=True)
    raw_img = Image.open(img_path)
    w0, h0 = raw_img.size
    print(f"size={w0}x{h0}", flush=True)

    ref_img = load_reference_patch(PANDA_REFERENCE)
    normalizer, backend = get_normalizer(args.method)
    normalizer.fit(ref_img)
    print(f"fit backend={backend} ref={PANDA_REFERENCE}", flush=True)

    stitched, worst_xy = stitch_normalized(raw_img, normalizer, tile=512, level=args.level)

    panda_slide_path = SLIDES_DIR / f"{PANDA_REFERENCE[0]}.tiff"
    panda_img = np.asarray(Image.open(panda_slide_path).convert("RGB"))
    print(f"panda_slide_size={panda_img.shape[1]}x{panda_img.shape[0]}", flush=True)

    raw_arr = np.asarray(raw_img.convert("RGB"))
    panels = [
        (thumb(raw_arr), f"AGGC raw\n{args.slide_id} ({w0}x{h0})"),
        (thumb(stitched), f"AGGC Macenko-stitched ({args.level}-level)\n(target=PANDA ref patch)"),
        (thumb(panda_img), f"PANDA reference slide\n{PANDA_REFERENCE[0][:10]}... (what we target)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}", flush=True)

    # Also save a tight crop at full res (no downsample) so tile seams are visible.
    # Centered on the tile with the biggest raw->normalized brightness drop
    # (a proxy for "this tile's per-tile stain estimate went bad"), not a guess.
    if worst_xy is not None:
        cy, cx = worst_xy[0] + 256, worst_xy[1] + 256
    else:
        cy, cx = raw_arr.shape[0] // 2, raw_arr.shape[1] // 2
    r = 1024
    crop_raw = raw_arr[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    crop_norm = stitched[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    axes2[0].imshow(crop_raw)
    axes2[0].set_title("raw (full-res crop)")
    axes2[0].axis("off")
    axes2[1].imshow(crop_norm)
    axes2[1].set_title("Macenko-stitched (full-res crop, tile seams visible here if present)")
    axes2[1].axis("off")
    fig2.tight_layout()
    crop_out = args.out.with_name(args.out.stem + "_fullres_crop.png")
    fig2.savefig(crop_out, dpi=150)
    print(f"saved {crop_out}", flush=True)


if __name__ == "__main__":
    main()
