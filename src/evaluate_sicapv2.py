"""Zero-shot SICAPv2 pixel eval for a PANDA-trained UNI2 Opt3 checkpoint.

Official SICAPv2 (Mendeley) masks use Gleason IDs {0, 3, 4, 5}:
  0 = non-cancerous, 3 = GG3, 4 = GG4, 5 = GG5
(README also documents 0/1/2/3; on-disk PNGs use 0/3/4/5.)

PANDA model predicts 6 classes {0..5}. For SICAP scoring we collapse
background/stroma/benign -> 0 (non-cancer) so the shared label space is
{0, 3, 4, 5}.

Reports per-class Dice, mean Dice over those 4, cancer Dice (mean G3/G4/G5),
and binary cancer Dice. Runs each Val fold Test.xlsx (patient-based CV).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
from train.sicap_metrics import (  # noqa: E402
    SICAP_CLASSES,
    SICAP_NAMES,
    accumulate_sicap_cm,
    dice_from_sicap_cm,
    remap_panda_pred,
)

DEFAULT_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")


def dice_from_cm(cm: np.ndarray, classes: tuple[int, ...] = SICAP_CLASSES) -> dict[str, float]:
    """Back-compat alias for ``dice_from_sicap_cm``."""
    return dice_from_sicap_cm(cm, classes=classes)


class SicapPatchDataset(Dataset):
    def __init__(self, root: Path, image_names: list[str]):
        self.root = root
        self.images = root / "images"
        self.masks = root / "masks"
        self.names: list[str] = []
        skipped = 0
        for name in image_names:
            stem = Path(name).stem
            # xlsx may store with or without .jpg
            img = self.images / f"{stem}.jpg"
            if not img.is_file():
                img = self.images / name
            msk = self.masks / f"{stem}.png"
            if not img.is_file() or not msk.is_file():
                skipped += 1
                continue
            self.names.append(stem)
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        stem = self.names[idx]
        img = np.array(Image.open(self.images / f"{stem}.jpg").convert("RGB"), dtype=np.uint8)
        msk = np.array(Image.open(self.masks / f"{stem}.png"))
        if msk.ndim == 3:
            msk = msk[..., 0]
        # float ImageNet-ish is NOT used in PANDA raw mode — check baseline
        # PANDA raw uses /255 then maybe nothing; peek baseline_dataset
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(msk.astype(np.int64))
        return x, y, stem


def load_split_names(root: Path, fold: int, split: str) -> list[str]:
    """split in {'Test','Train'}."""
    path = root / "partition" / "Validation" / f"Val{fold}" / f"{split}.xlsx"
    df = pd.read_excel(path)
    col = "image_name" if "image_name" in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def load_test_names(root: Path, fold: int) -> list[str]:
    return load_split_names(root, fold, "Test")


def load_official_test_names(root: Path) -> list[str]:
    path = root / "partition" / "Test" / "Test.xlsx"
    df = pd.read_excel(path)
    col = "image_name" if "image_name" in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


@torch.no_grad()
def predict_batch_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    scale_factor: float,
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """Logits (B,C,H,W) at original resolution. scale>1: upsample→512-tile→stitch→down."""
    import torch.nn.functional as F

    images = images.to(device, non_blocking=True)
    if abs(scale_factor - 1.0) < 1e-6:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            return model(images)

    b, _, h0, w0 = images.shape
    up = F.interpolate(
        images, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )
    _, _, h, w = up.shape
    ps = 512
    if h % ps != 0 or w % ps != 0:
        ph = (ps - h % ps) % ps
        pw = (ps - w % ps) % ps
        up = F.pad(up, (0, pw, 0, ph), mode="reflect")
        _, _, h, w = up.shape

    tiles = []
    coords = []
    for y in range(0, h, ps):
        for x in range(0, w, ps):
            tiles.append(up[:, :, y : y + ps, x : x + ps])
            coords.append((y, x))
    tile_batch = torch.cat(tiles, dim=0)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        tile_logits = model(tile_batch)
    c = tile_logits.shape[1]
    canvas = torch.zeros(b, c, h, w, device=device, dtype=tile_logits.dtype)
    for ti, (y, x) in enumerate(coords):
        canvas[:, :, y : y + ps, x : x + ps] = tile_logits[ti * b : (ti + 1) * b]
    h_keep = min(h, int(round(h0 * scale_factor)))
    w_keep = min(w, int(round(w0 * scale_factor)))
    canvas = canvas[:, :, :h_keep, :w_keep]
    return F.interpolate(canvas, size=(h0, w0), mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    scale_factor: float,
    amp: bool,
    device: torch.device,
) -> np.ndarray:
    """Argmax preds at original HxW."""
    logits = predict_batch_logits(
        model, images, scale_factor=scale_factor, amp=amp, device=device
    )
    return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def eval_named_split(
    model: torch.nn.Module,
    device: torch.device,
    root: Path,
    names: list[str],
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
    scale_factor: float = 1.0,
    fold_label: int | str = 0,
) -> dict:
    ds = SicapPatchDataset(root, names)
    bs = 1 if scale_factor > 1.0 else batch_size
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    cm = np.zeros((6, 6), dtype=np.int64)
    n_pix = 0
    for images, masks, _stems in loader:
        pred = predict_batch(
            model, images, scale_factor=scale_factor, amp=amp, device=device
        )
        gt = masks.numpy()
        accumulate_sicap_cm(cm, pred, gt)
        n_pix += int(np.asarray(gt).size)
    metrics = dice_from_cm(cm, SICAP_CLASSES)
    return {
        "fold": fold_label,
        "n_listed": len(names),
        "n_scored": len(ds),
        "n_skipped": ds.skipped,
        "n_pixels": int(n_pix),
        "scale_factor": float(scale_factor),
        **metrics,
        "confusion_0_3_4_5": {
            f"t{t}_p{p}": int(cm[t, p])
            for t in SICAP_CLASSES
            for p in SICAP_CLASSES
        },
    }


@torch.no_grad()
def eval_fold(
    model: torch.nn.Module,
    device: torch.device,
    root: Path,
    fold: int,
    batch_size: int,
    num_workers: int,
    amp: bool,
    scale_factor: float = 1.0,
) -> dict:
    return eval_named_split(
        model,
        device,
        root,
        load_test_names(root, fold),
        batch_size=batch_size,
        num_workers=num_workers,
        amp=amp,
        scale_factor=scale_factor,
        fold_label=fold,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="SICAPv2 zero-shot eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--folds", type=str, default="1,2,3,4", help="Comma folds, e.g. 1 or 1,2,3,4")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help=(
            "Physical-scale align vs PANDA train (~0.49 mpp / 20×). "
            "SICAPv2 10× ≈2× coarser → use 2.0: upsample, 512-tile, stitch, "
            "downsample pred to native 512 GT."
        ),
    )
    p.add_argument(
        "--partition",
        choices=["val", "official"],
        default="val",
        help="val=Validation/ValN/Test.xlsx (CV). official=partition/Test/Test.xlsx holdout.",
    )
    args = p.parse_args()

    root = args.sicap_root
    if not (root / "images").is_dir() or not (root / "masks").is_dir():
        raise SystemExit(f"SICAPv2 images/masks missing under {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for SICAPv2 eval")

    resolved = detect_arch(args.checkpoint, "auto")
    _ckpt, keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()
    print(
        f"ckpt={args.checkpoint} arch={resolved} decode_norm={decode_norm} "
        f"lora={use_lora} scale_factor={args.scale_factor}",
        flush=True,
    )

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    rows = []
    if args.partition == "official":
        print("=== official partition/Test/Test.xlsx ===", flush=True)
        row = eval_named_split(
            model,
            device,
            root,
            load_official_test_names(root),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            scale_factor=args.scale_factor,
            fold_label="official",
        )
        rows.append(row)
        print(
            f"  scored={row['n_scored']}/{row['n_listed']}  "
            f"cancer={row['cancer']:.4f} mean4={row['mean4']:.4f} "
            f"bin={row['binary_cancer']:.4f}  "
            f"NC={row['NC']:.4f} G3={row['G3']:.4f} G4={row['G4']:.4f} G5={row['G5']:.4f} "
            f"nc_to_ca={row.get('nc_to_cancer', float('nan')):.4f}",
            flush=True,
        )
    else:
        for fold in folds:
            print(f"=== Val{fold} Test ===", flush=True)
            row = eval_fold(
                model,
                device,
                root,
                fold,
                args.batch_size,
                args.num_workers,
                args.amp,
                scale_factor=args.scale_factor,
            )
            rows.append(row)
            print(
                f"  scored={row['n_scored']}/{row['n_listed']}  "
                f"cancer={row['cancer']:.4f} mean4={row['mean4']:.4f} "
                f"bin={row['binary_cancer']:.4f}  "
                f"NC={row['NC']:.4f} G3={row['G3']:.4f} G4={row['G4']:.4f} G5={row['G5']:.4f}",
                flush=True,
            )

    # mean across folds
    scale = float(args.scale_factor)
    protocol = "remap_panda_bg_stroma_benign_to_NC; GT labels {0,3,4,5}"
    if scale != 1.0:
        protocol += (
            f"; mpp_align scale_factor={scale} "
            "(upsample→512-tile→stitch→downsample pred to native GT)"
        )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sicap_root": str(root.resolve()),
        "folds": folds if args.partition == "val" else ["official"],
        "partition": args.partition,
        "scale_factor": scale,
        "per_fold": rows,
        "mean_cancer_dice": float(np.mean([r["cancer"] for r in rows])),
        "mean_mean4_dice": float(np.mean([r["mean4"] for r in rows])),
        "mean_binary_cancer_dice": float(np.mean([r["binary_cancer"] for r in rows])),
        "protocol": protocol,
    }
    print(
        f"MEAN folds cancer={summary['mean_cancer_dice']:.4f} "
        f"mean4={summary['mean_mean4_dice']:.4f} "
        f"binary={summary['mean_binary_cancer_dice']:.4f} "
        f"scale={scale}",
        flush=True,
    )

    stem = args.checkpoint.stem
    if args.partition == "official":
        stem = f"official_{stem}"
    if scale != 1.0:
        stem = f"mpp{scale:g}x_{stem}"
    out_dir = args.out_dir or (OUTPUTS / "evaluation" / "sicapv2" / stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sicapv2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "sicapv2_per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold", "n_scored", "n_listed", "n_skipped",
                "NC", "G3", "G4", "G5", "mean4", "cancer", "binary_cancer",
                "nc_to_cancer",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
