"""Diagnostic: does closing / hole-fill of predicted cancer raise SICAPv2 Dice?

Reads NO training state. One inference pass per checkpoint; every post-proc mode
is scored on the same argmax maps, so mode vs base is an exact comparison.

Meaning of the result
  * Dice rises by >= ~0.05 with a modest radius  -> GT masks are coarse polygons and
    the model's patchy predictions are penalised for fine detail the labels never
    had; closing/fill is a legitimate post-process (apply identically at test time).
  * Dice flat or falling                         -> the model really under-segments
    (extent error); post-processing cannot fix it, training signal must.

Modes (cancer = predicted class >= 3; cancer pixels are never removed):
  base       argmax as-is
  fill       fill enclosed non-cancer holes
  cR         morphological close (elliptical, radius R px) then fill, R in {3,7,15,31}
New cancer pixels take the grade of the nearest original cancer pixel.

No rescaling of the image is performed anywhere (native 512 tiles, scale 1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    detect_arch,
    load_model_weights,
)
from evaluate_sicapv2 import (  # noqa: E402
    DEFAULT_ROOT,
    SicapPatchDataset,
    dice_from_cm,
    load_official_test_names,
    load_test_names,
    predict_batch,
)
from train.sicap_metrics import SICAP_CLASSES, accumulate_sicap_cm  # noqa: E402

MODES = ("base", "fill", "c3", "c7", "c15", "c31")


def close_cancer(pred: np.ndarray, mode: str) -> np.ndarray:
    """pred: (H,W) int PANDA ids 0..5. Returns the post-processed copy."""
    if mode == "base":
        return pred
    import cv2
    from scipy import ndimage as ndi

    cancer = pred >= 3
    if not cancer.any():
        return pred
    m = cancer
    if mode.startswith("c"):
        r = int(mode[1:])
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    m = ndi.binary_fill_holes(m)
    added = m & ~cancer
    if not added.any():
        return pred
    # nearest original cancer pixel -> its grade
    _, (iy, ix) = ndi.distance_transform_edt(~cancer, return_indices=True)
    out = pred.copy()
    out[added] = pred[iy[added], ix[added]]
    return out


def pick_best_checkpoint(run_dir: Path) -> Path:
    """Val-selected epoch = highest cancer value in the epoch_XXX_cancer_Y.pth name."""
    best, best_v = None, -1.0
    for p in sorted(run_dir.glob("epoch_*_cancer_*.pth")):
        try:
            v = float(p.stem.split("_cancer_")[1])
        except (IndexError, ValueError):
            continue
        if v > best_v:
            best, best_v = p, v
    if best is None:
        raise SystemExit(f"no epoch_*_cancer_*.pth in {run_dir}")
    return best


@torch.no_grad()
def sweep(model, device, root, names, *, batch_size, num_workers, amp) -> dict:
    ds = SicapPatchDataset(root, names)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cms = {m: np.zeros((6, 6), dtype=np.int64) for m in MODES}
    added_px = {m: 0 for m in MODES}
    for images, masks, _stems in loader:
        pred = predict_batch(model, images, scale_factor=1.0, amp=amp, device=device)
        gt = masks.numpy()
        for b in range(pred.shape[0]):
            for m in MODES:
                p2 = close_cancer(pred[b], m)
                added_px[m] += int(((p2 >= 3) & (pred[b] < 3)).sum())
                accumulate_sicap_cm(cms[m], p2[None], gt[b][None])
    out = {"n_scored": len(ds), "modes": {}}
    for m in MODES:
        d = dice_from_cm(cms[m], SICAP_CLASSES)
        d["added_cancer_px"] = added_px[m]
        out["modes"][m] = d
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--run-dir", type=Path, help="pick Val-best epoch_*_cancer_*.pth")
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--partition", choices=["val", "official"], default="val")
    p.add_argument("--fold", type=int, default=1, help="Val fold (partition=val)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    ckpt = args.checkpoint or pick_best_checkpoint(args.run_dir)
    device = torch.device("cuda")
    resolved = detect_arch(ckpt, "auto")
    _c, keys = _peek_state_dict(ckpt)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(ckpt, model)
    model.to(device).eval()

    names = (
        load_official_test_names(args.sicap_root)
        if args.partition == "official"
        else load_test_names(args.sicap_root, args.fold)
    )
    res = sweep(
        model, device, args.sicap_root, names,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
    )
    res.update(
        label=args.label, checkpoint=str(ckpt), partition=args.partition,
        fold=args.fold if args.partition == "val" else "official",
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2))
    base = res["modes"]["base"]["cancer"]
    print(f"[{args.label} {args.partition} f{args.fold}] ckpt={ckpt.name} n={res['n_scored']}")
    for m in MODES:
        d = res["modes"][m]
        print(
            f"  {m:5s} cancer={d['cancer']:.4f} ({d['cancer']-base:+.4f}) "
            f"bin={d['binary_cancer']:.4f} G3={d['G3']:.4f} G4={d['G4']:.4f} "
            f"G5={d['G5']:.4f} nc_to_ca={d.get('nc_to_cancer', float('nan')):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
