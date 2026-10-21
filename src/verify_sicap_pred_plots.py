"""Check that SICAP prediction pictures show what the scorer scores, for two checkpoints.

1. Full official-test scan per checkpoint with the same SICAP confusion matrix as
   ``evaluate_sicapv2``. Printed next to the saved official summary.
2. Batched vs single-tile argmax on every example tile (must be identical).
3. Per tile: raw label PNGs plus a 2x4 figure. Top row is flat class colors
   (tissue, mask, model A, model B). Bottom row is the same labels over tissue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image
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
    SicapPatchDataset,
    load_official_test_names,
    predict_batch,
)
from render_sicap_pred_vs_mask import SOLID_COLORS, colorize, sheet_grade  # noqa: E402
from train.sicap_metrics import accumulate_sicap_cm, dice_from_sicap_cm  # noqa: E402

DEFAULT_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")
CLASS_NAMES = ["background", "stroma", "benign", "G3", "G4", "G5"]


def load(ckpt: Path, device: torch.device) -> torch.nn.Module:
    arch = detect_arch(ckpt, "auto")
    _c, keys = _peek_state_dict(ckpt)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(arch, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(ckpt, model)
    return model.to(device).eval()


def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = rgb.astype(np.float32)
    for cls, color in SOLID_COLORS.items():
        if cls == 0:
            continue
        m = labels == cls
        out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def class_share(labels: np.ndarray) -> str:
    counts = np.bincount(labels.ravel(), minlength=6) / labels.size
    parts = [f"{CLASS_NAMES[c]} {counts[c] * 100:.0f}%" for c in range(6) if counts[c] >= 0.01]
    return ", ".join(parts)


def scan(model, device, root: Path, batch_size: int) -> dict:
    ds = SicapPatchDataset(root, load_official_test_names(root))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    cm = np.zeros((6, 6), dtype=np.int64)
    batched: dict[str, np.ndarray] = {}
    for images, masks, stems in loader:
        pred = predict_batch(model, images, scale_factor=1.0, amp=True, device=device)
        accumulate_sicap_cm(cm, pred, masks.numpy())
        for i, s in enumerate(stems):
            batched[str(s)] = pred[i].astype(np.uint8)
    m = dice_from_sicap_cm(cm)
    nc_row = cm[0].sum()
    m["nc_to_cancer"] = float(cm[0, 3:].sum() / nc_row) if nc_row else float("nan")
    m["n_scored"] = len(ds)
    return {"metrics": m, "batched": batched}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", type=Path, required=True)
    p.add_argument("--name-a", type=str, required=True)
    p.add_argument("--summary-a", type=Path, required=True)
    p.add_argument("--ckpt-b", type=Path, required=True)
    p.add_argument("--name-b", type=str, required=True)
    p.add_argument("--summary-b", type=Path, required=True)
    p.add_argument("--stems", type=str, required=True, help="Pipe-separated official tile stems")
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    stems = [s for s in args.stems.split("|") if s]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"models": {}, "tiles": []}
    preds: dict[str, dict[str, np.ndarray]] = {}

    for name, ckpt, summary in (
        (args.name_a, args.ckpt_a, args.summary_a),
        (args.name_b, args.ckpt_b, args.summary_b),
    ):
        model = load(ckpt, device)
        res = scan(model, device, args.sicap_root, args.batch_size)
        saved = json.loads(summary.read_text())["per_fold"][0]
        keys = ("cancer", "mean4", "NC", "G3", "G4", "G5", "nc_to_cancer")
        report["models"][name] = {
            "checkpoint": str(ckpt),
            "rescan": {k: float(res["metrics"][k]) for k in keys},
            "saved_official": {k: float(saved[k]) for k in keys},
        }
        print(f"[{name}] {ckpt}", flush=True)
        for k in keys:
            print(
                f"  {k:13s} rescan={res['metrics'][k]:.4f} saved={saved[k]:.4f}",
                flush=True,
            )

        ds = SicapPatchDataset(args.sicap_root, stems)
        preds[name] = {}
        for i, stem in enumerate(ds.names):
            x, _y, _s = ds[i]
            single = predict_batch(
                model, x.unsqueeze(0), scale_factor=1.0, amp=True, device=device
            )[0].astype(np.uint8)
            same = float((single == res["batched"][stem]).mean())
            print(f"  {stem}: single vs batched agree {same * 100:.3f}%", flush=True)
            preds[name][stem] = single
            Image.fromarray(single).save(args.out_dir / f"{stem}__pred_{name}.png")
        del model
        torch.cuda.empty_cache()

    ds = SicapPatchDataset(args.sicap_root, stems)
    for i, stem in enumerate(ds.names):
        x, y, _s = ds[i]
        rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = y.numpy().astype(np.uint8)
        pa = preds[args.name_a][stem]
        pb = preds[args.name_b][stem]
        grade = sheet_grade(args.sicap_root, stem)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8.6), dpi=130)
        top = [
            (rgb, "tissue"),
            (colorize(gt), f"provided mask\n{class_share(gt)}"),
            (colorize(pa), f"{args.name_a}\n{class_share(pa)}"),
            (colorize(pb), f"{args.name_b}\n{class_share(pb)}"),
        ]
        bottom = [
            (rgb, "tissue"),
            (overlay(rgb, gt), "mask over tissue"),
            (overlay(rgb, pa), f"{args.name_a} over tissue"),
            (overlay(rgb, pb), f"{args.name_b} over tissue"),
        ]
        for ax, (img, title) in zip(list(axes[0]) + list(axes[1]), top + bottom):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
        fig.suptitle(f"sheet grade {grade}  |  {stem}", fontsize=11)
        handles = [
            Patch(facecolor=np.array(SOLID_COLORS[0]) / 255, edgecolor="0.6", label="background / mask non-cancer"),
            Patch(facecolor=np.array(SOLID_COLORS[1]) / 255, label="stroma"),
            Patch(facecolor=np.array(SOLID_COLORS[2]) / 255, label="benign"),
            Patch(facecolor=np.array(SOLID_COLORS[3]) / 255, label="G3"),
            Patch(facecolor=np.array(SOLID_COLORS[4]) / 255, label="G4"),
            Patch(facecolor=np.array(SOLID_COLORS[5]) / 255, label="G5"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        out = args.out_dir / f"{grade}_{stem}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        Image.fromarray(gt).save(args.out_dir / f"{stem}__mask.png")
        report["tiles"].append(
            {
                "stem": stem,
                "sheet": grade,
                "mask": class_share(gt),
                args.name_a: class_share(pa),
                args.name_b: class_share(pb),
                "png": str(out),
            }
        )
        print(f"wrote {out}", flush=True)

    (args.out_dir / "verify.json").write_text(json.dumps(report, indent=2))
    print(f"OK {args.out_dir / 'verify.json'}", flush=True)


if __name__ == "__main__":
    main()
