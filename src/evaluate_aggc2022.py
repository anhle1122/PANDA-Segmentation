"""Zero-shot AGGC2022 eval for a PANDA/SICAP UNI2 Opt3 checkpoint.

AGGC2022 (https://aggc22.grand-challenge.org/) ships per-class binary masks
(Stroma/Normal/G3/G4/G5, any subset present) and scores a single "index
image" per class 0=background(unannotated) 1=Stroma 2=Normal 3=G3 4=G4 5=G5
(see https://github.com/jydada/AGGC2022, eval.py). Our model's 6 output
channels already use that exact ordering, so argmax needs no remap.

Pipeline per slide:
  1. Build GT index map from raw per-class masks: paint Stroma, then Normal,
     then G3, G4, G5 in that order (last write wins) -> "most malignant wins"
     on overlap, matching AGGC's own documented convention (their one
     deliberate-nested-lower-grade exception is not reproduced here).
  2. Resample the H&E image to the inference scale (native, or a target
     µm/pixel) with bilinear, tile at --tile-size, batch through the model,
     argmax per tile (no remap needed, see above), stitch, then resize the
     prediction map back up to the image's NATIVE resolution (nearest, it's
     categorical) so it lines up pixel-for-pixel with the GT map above.
  3. Per AGGC's own eval spec, downsize BOTH the GT and prediction maps by
     10x (nearest, ceil on non-integer sizes) -> "2x index image" -- this is
     a fixed scoring-resolution step, independent of the inference scale in
     step 2.
  4. Accumulate one 6x6 confusion matrix per (subset, split, scale) group
     across all its slides (never pooled across subsets) and compute the
     official weighted F1 with a near-verbatim port of their eval.py. F1 and
     per-class Dice are the same formula (2*TP/(2*TP+FP+FN)), so this is
     already a per-class Dice report, just combined into one weighted score.

Subset 1/2 images are Akoya-scanned (manual annotation is Akoya-only); we use
one fixed native µm/pixel for both. Subset 3 (multi-scanner, registered
masks) is out of scope for this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    build_eval_model,
    detect_arch,
    load_model_weights,
    _opt3_flags,
    _peek_state_dict,
)
from patch_utils import OUTPUTS  # noqa: E402

DEFAULT_ROOT = Path("/common/omarmlab/data/AGGC_2022")
AKOYA_MPP = 0.500553  # manual annotation scanner; Subset1/2 images are Akoya-scanned.

# Paint order = ascending malignancy; last write wins -> most malignant wins.
MASK_PRIORITY = [
    ("Stroma_Mask.tif", 1),
    ("Normal_Mask.tif", 2),
    ("G3_Mask.tif", 3),
    ("G4_Mask.tif", 4),
    ("G5_Mask.tif", 5),
]
CLASS_NAMES = {0: "background", 1: "Stroma", 2: "Normal", 3: "G3", 4: "G4", 5: "G5"}
# AGGC's own weighted-average F1 (== per-class Dice, same formula).
CLASS_WEIGHTS = {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.25, 5: 0.25}


def list_slides(root: Path, subset: str, split: str) -> list[str]:
    img_dir = root / subset / split / "images"
    ann_dir = root / subset / split / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        return []
    have_img = {p.stem for p in img_dir.glob("*.tiff")} | {p.stem for p in img_dir.glob("*.tif")}
    have_ann = {p.name for p in ann_dir.iterdir() if p.is_dir()}
    names = sorted(have_img & have_ann)
    return names


def find_image_path(img_dir: Path, slide_id: str) -> Path | None:
    for ext in (".tiff", ".tif"):
        p = img_dir / f"{slide_id}{ext}"
        if p.is_file():
            return p
    return None


def build_gt_index_map(ann_dir: Path, slide_id: str, shape: tuple[int, int]) -> np.ndarray:
    """Resolve raw per-class binary masks into one 0..5 index map (most-malignant-wins)."""
    gt = np.zeros(shape, dtype=np.uint8)
    slide_dir = ann_dir / slide_id
    for fname, class_idx in MASK_PRIORITY:
        mpath = slide_dir / fname
        if not mpath.is_file():
            continue
        m = np.array(Image.open(mpath))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != shape:
            m = np.array(
                Image.fromarray(m).resize((shape[1], shape[0]), resample=Image.NEAREST)
            )
        gt[m > 0] = class_idx
    return gt


@torch.no_grad()
def infer_index_map(
    model: torch.nn.Module,
    img: Image.Image,
    *,
    device: torch.device,
    amp: bool,
    scale_factor: float,
    tile: int = 512,
    batch_tiles: int = 8,
) -> np.ndarray:
    """Full-slide tiled inference -> argmax index map at the image's NATIVE resolution.

    scale_factor applies to the INPUT before tiling (e.g. 0.5 to go from
    ~20x/0.5um toward SICAP's ~10x/1.0um). The output is resized back up to
    img's native (W, H) with nearest-neighbor so it aligns with the GT map.
    """
    w0, h0 = img.size
    if abs(scale_factor - 1.0) > 1e-6:
        w1, h1 = max(1, round(w0 * scale_factor)), max(1, round(h0 * scale_factor))
        resized = img.resize((w1, h1), resample=Image.BILINEAR)
    else:
        w1, h1 = w0, h0
        resized = img
    arr = np.asarray(resized.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    ph = (tile - h % tile) % tile
    pw = (tile - w % tile) % tile
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    hp, wp = arr.shape[:2]

    pred = np.zeros((hp, wp), dtype=np.uint8)
    all_coords = [(y, x) for y in range(0, hp, tile) for x in range(0, wp, tile)]
    # Whole-mount (Subset1) slides are mostly scanner background -> cheap tissue
    # check (mean/std of raw RGB) skips blank-white tiles without a model forward.
    # Any tile that passes is still argmax'd by the model as usual (bg class 0 is
    # a normal model output), so this only saves compute, it never forces a label.
    coords = []
    n_skipped = 0
    for y, x in all_coords:
        t = arr[y : y + tile, x : x + tile]
        if t.mean() > 220.0 and t.std() < 12.0:
            n_skipped += 1
            continue
        coords.append((y, x))

    for i in range(0, len(coords), batch_tiles):
        batch_coords = coords[i : i + batch_tiles]
        batch = np.stack(
            [arr[y : y + tile, x : x + tile] for y, x in batch_coords], axis=0
        )
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device) / 255.0
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        argmax = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
        for (y, x0), am in zip(batch_coords, argmax):
            pred[y : y + tile, x0 : x0 + tile] = am
    print(
        f"    tiles total={len(all_coords)} skipped_bg={n_skipped} "
        f"ran={len(coords)}",
        flush=True,
    )

    pred = pred[:h, :w]
    if (h, w) != (h0, w0):
        pred = np.array(
            Image.fromarray(pred).resize((w0, h0), resample=Image.NEAREST)
        )
    return pred


def downsize_10x(index_map: np.ndarray) -> np.ndarray:
    """AGGC's fixed scoring step: resize to '2x' = downsize original by 10x, ceil, nearest."""
    h, w = index_map.shape
    h2 = -(-h // 10)  # ceil
    w2 = -(-w // 10)
    return np.array(
        Image.fromarray(index_map).resize((w2, h2), resample=Image.NEAREST)
    )


def weighted_f1_from_cm(cm: np.ndarray) -> dict:
    """Near-verbatim port of jydada/AGGC2022 eval.py. cm[gt, pred], 6x6."""
    out: dict = {}
    f1s: dict[int, float] = {}
    for c in (1, 2, 3, 4, 5):
        tp = cm[c, c]
        recall_den = cm[c, :].sum()
        # Precision denominator excludes predictions landing on gt==0 (unannotated/background):
        # AGGC only scores pixels inside annotated regions, so a prediction there is forgiven.
        prec_den = cm[:, c].sum() - cm[0, c]
        recall = float(tp) / recall_den if recall_den else float("nan")
        precision = float(tp) / prec_den if prec_den else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) and not (np.isnan(precision) or np.isnan(recall))
            else float("nan")
        )
        f1s[c] = f1
        out[f"recall_{CLASS_NAMES[c]}"] = recall
        out[f"precision_{CLASS_NAMES[c]}"] = precision
        out[f"dice_{CLASS_NAMES[c]}"] = f1  # F1 == Dice for this confusion-matrix formula.
    valid = [CLASS_WEIGHTS[c] * f1s[c] for c in CLASS_WEIGHTS if not np.isnan(f1s[c])]
    wsum = sum(CLASS_WEIGHTS[c] for c in CLASS_WEIGHTS if not np.isnan(f1s[c]))
    out["weighted_f1"] = float(sum(valid) / wsum) if wsum else float("nan")
    nc_mass = cm[1, :].sum() + cm[2, :].sum()
    nc_to_cancer = float(cm[1, 3:6].sum() + cm[2, 3:6].sum()) / nc_mass if nc_mass else float("nan")
    out["nc_to_cancer"] = nc_to_cancer
    return out


def run_group(
    model: torch.nn.Module,
    device: torch.device,
    root: Path,
    subset: str,
    split: str,
    *,
    scale_factor: float,
    amp: bool,
    tile: int,
    batch_tiles: int,
    max_slides: int,
    out_dir: Path,
) -> dict:
    img_dir = root / subset / split / "images"
    ann_dir = root / subset / split / "annotations"
    names = list_slides(root, subset, split)
    if max_slides:
        names = names[:max_slides]
    cm = np.zeros((6, 6), dtype=np.int64)
    rows = []
    t0 = time.time()
    for i, sid in enumerate(names):
        img_path = find_image_path(img_dir, sid)
        if img_path is None:
            print(f"  WARN missing image for {sid}", flush=True)
            continue
        img = Image.open(img_path)
        w0, h0 = img.size
        pred = infer_index_map(
            model, img, device=device, amp=amp, scale_factor=scale_factor,
            tile=tile, batch_tiles=batch_tiles,
        )
        gt = build_gt_index_map(ann_dir, sid, (h0, w0))
        gt2 = downsize_10x(gt)
        pred2 = downsize_10x(pred)
        for c_gt in range(6):
            mask_gt = gt2 == c_gt
            if not mask_gt.any():
                continue
            row_pred = pred2[mask_gt]
            counts = np.bincount(row_pred, minlength=6)
            cm[c_gt, :] += counts
        rows.append({"slide_id": sid, "w": w0, "h": h0})
        elapsed = time.time() - t0
        print(
            f"  [{i + 1}/{len(names)}] {sid} {w0}x{h0} elapsed={elapsed:.0f}s",
            flush=True,
        )
    metrics = weighted_f1_from_cm(cm)
    out = {
        "subset": subset,
        "split": split,
        "scale_factor": scale_factor,
        "n_slides": len(rows),
        "n_listed": len(names),
        **metrics,
        "confusion_matrix": cm.tolist(),
    }
    tag_dir = out_dir / f"{subset}_{split}"
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "summary.json").write_text(json.dumps(out, indent=2))
    with (tag_dir / "slides.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id", "w", "h"])
        w.writeheader()
        w.writerows(rows)
    print(
        f"=== {subset}/{split} n={out['n_slides']} weighted_f1={out['weighted_f1']:.4f} "
        f"NC->cancer={out['nc_to_cancer']:.4f} ===",
        flush=True,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="AGGC2022 zero-shot eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--aggc-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--groups",
        type=str,
        default="Subset1/Test,Subset2/Train,Subset2/Test",
        help="Comma list of Subset{1,2}/{Train,Test}. Never pooled across groups.",
    )
    p.add_argument(
        "--scale-mode",
        choices=["native", "to10x"],
        default="native",
        help="native = no resample (~Akoya 0.5um/px as-is). to10x = resample toward SICAP's ~1.0um/px.",
    )
    p.add_argument("--native-mpp", type=float, default=AKOYA_MPP)
    p.add_argument("--target-mpp", type=float, default=1.0)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--batch-tiles", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-slides", type=int, default=0, help="0 = all; >0 for a smoke test.")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for AGGC2022 eval")

    resolved = detect_arch(args.checkpoint, "auto")
    _ckpt, keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()

    scale_factor = args.native_mpp / args.target_mpp if args.scale_mode == "to10x" else 1.0
    print(
        f"ckpt={args.checkpoint} arch={resolved} decode_norm={decode_norm} lora={use_lora} "
        f"scale_mode={args.scale_mode} scale_factor={scale_factor:.4f} groups={args.groups}",
        flush=True,
    )

    out_dir = args.out_dir or (
        OUTPUTS / "evaluation" / "aggc2022" / f"{args.checkpoint.stem}_{args.scale_mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for group in args.groups.split(","):
        subset, split = group.strip().split("/")
        print(f"=== {subset}/{split} ===", flush=True)
        results.append(
            run_group(
                model, device, args.aggc_root, subset, split,
                scale_factor=scale_factor, amp=args.amp, tile=args.tile_size,
                batch_tiles=args.batch_tiles, max_slides=args.max_slides, out_dir=out_dir,
            )
        )
    (out_dir / "all_groups.json").write_text(json.dumps(results, indent=2))
    print(f"=== DONE out_dir={out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
