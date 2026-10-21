"""Render SICAP tiles where a checkpoint calls non-cancer tissue cancer, or misses a cancer region.

Official-test scan. Predictions stay 6-class in the picture. The error
counts collapse background/stroma/benign to non-cancer, same as SICAP scoring.
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
from train.sicap_slide_bag import slide_id_from_patch  # noqa: E402

PROJECT = SRC.parent
DEFAULT_CKPT = (
    PROJECT
    / "outputs/checkpoints/uni2_upernet_raw_opt3_omar6_round3_ep7ref"
    / "epoch_006_cancer_0.3488.pth"
)
DEFAULT_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")

CLASS_NAMES = {
    0: "background",
    1: "stroma",
    2: "benign",
    3: "G3",
    4: "G4",
    5: "G5",
}
CLASS_COLORS = {
    1: (30, 90, 220),
    2: (20, 160, 60),
    3: (240, 220, 40),
    4: (240, 140, 20),
    5: (210, 30, 30),
}
BORDER = 24


SOLID_COLORS = {
    0: (245, 245, 245),
    1: (30, 90, 220),
    2: (20, 160, 60),
    3: (240, 220, 40),
    4: (240, 140, 20),
    5: (210, 30, 30),
}


def colorize(labels: np.ndarray) -> np.ndarray:
    """Flat class colors. No tissue under the mask or the prediction."""
    out = np.empty(labels.shape + (3,), dtype=np.uint8)
    out[:] = SOLID_COLORS[0]
    for cls, color in SOLID_COLORS.items():
        if cls == 0:
            continue
        out[labels == cls] = color
    return out


def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32)
    for cls, color in CLASS_COLORS.items():
        m = labels == cls
        if not np.any(m):
            continue
        out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def interior_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = pred[BORDER:-BORDER, BORDER:-BORDER]
    g = gt[BORDER:-BORDER, BORDER:-BORDER]
    n = int(p.size)
    nc = g == 0
    cancer = g >= 3
    pred_cancer = p >= 3
    fp = nc & pred_cancer
    fn = cancer & (p <= 2)
    n_nc = int(nc.sum())
    n_ca = int(cancer.sum())
    fp_hist = np.bincount(p[fp], minlength=6) if fp.any() else np.zeros(6, dtype=np.int64)
    fn_hist = np.bincount(p[fn], minlength=6) if fn.any() else np.zeros(6, dtype=np.int64)
    return {
        "n": n,
        "nc": n_nc,
        "gt_cancer": n_ca,
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "fp_grade": int(fp_hist[3:].argmax() + 3) if fp.any() else -1,
        "fn_class": int(fn_hist[:3].argmax()) if fn.any() else -1,
        "fp_of_nc": float(fp.sum() / n_nc) if n_nc else 0.0,
        "fn_of_cancer": float(fn.sum() / n_ca) if n_ca else 0.0,
        "nc_frac": float(n_nc / n) if n else 0.0,
        "gt_cancer_frac": float(n_ca / n) if n else 0.0,
        "fp_frac": float(fp.sum() / n) if n else 0.0,
    }


def pick(rows: list[dict], kind: str, k: int) -> list[dict]:
    if kind == "fp":
        pool = [
            r
            for r in rows
            if r["nc_frac"] >= 0.60
            and r["gt_cancer_frac"] <= 0.08
            and r["fp_of_nc"] >= 0.25
            and r["fp_frac"] >= 0.15
        ]
        pool.sort(key=lambda r: r["fp_of_nc"], reverse=True)
    else:
        pool = [
            r
            for r in rows
            if r["gt_cancer_frac"] >= 0.35 and r["fn_of_cancer"] >= 0.50
        ]
        pool.sort(key=lambda r: (r["fn_of_cancer"], r["gt_cancer_frac"]), reverse=True)
    chosen: list[dict] = []
    seen: set[str] = set()
    for r in pool:
        if r["slide"] in seen:
            continue
        seen.add(r["slide"])
        chosen.append(r)
        if len(chosen) >= k:
            break
    return chosen


def render_separate(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
    """Tissue, mask, and prediction in three panels. Nothing is painted on top of anything else."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), dpi=140)
    panels = [
        (rgb, "tissue"),
        (colorize(gt), "provided mask"),
        (colorize(pred), "prediction"),
    ]
    for ax, (img, name) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    handles = [
        Patch(facecolor=np.array(SOLID_COLORS[0]) / 255, edgecolor="0.6", label="non-cancer / background"),
        Patch(facecolor=np.array(SOLID_COLORS[1]) / 255, label="stroma"),
        Patch(facecolor=np.array(SOLID_COLORS[2]) / 255, label="benign"),
        Patch(facecolor=np.array(SOLID_COLORS[3]) / 255, label="G3"),
        Patch(facecolor=np.array(SOLID_COLORS[4]) / 255, label="G4"),
        Patch(facecolor=np.array(SOLID_COLORS[5]) / 255, label="G5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_row(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=140)
    panels = [
        (rgb, "tissue"),
        (overlay(rgb, gt), "mask"),
        (overlay(rgb, pred), "prediction"),
    ]
    for ax, (img, name) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    handles = [
        Patch(facecolor="white", edgecolor="0.6", label="background / mask non-cancer"),
        Patch(facecolor=np.array(CLASS_COLORS[1]) / 255, label="stroma"),
        Patch(facecolor=np.array(CLASS_COLORS[2]) / 255, label="benign"),
        Patch(facecolor=np.array(CLASS_COLORS[3]) / 255, label="G3"),
        Patch(facecolor=np.array(CLASS_COLORS[4]) / 255, label="G4"),
        Patch(facecolor=np.array(CLASS_COLORS[5]) / 255, label="G5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sheet_grade(root: Path, stem: str) -> str:
    import pandas as pd

    df = pd.read_excel(root / "partition" / "Test" / "Test.xlsx")
    df["stem"] = df["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    hit = df[df["stem"] == stem]
    if hit.empty:
        return "unlisted"
    row = hit.iloc[0]
    for name in ("NC", "G3", "G4", "G5"):
        if int(row[name]) == 1:
            return name
    return "unlisted"


def render_named(model, device, args) -> None:
    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    ds = SicapPatchDataset(args.sicap_root, stems)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, stem in enumerate(ds.names):
        x, y, _name = ds[i]
        pred = predict_batch(
            model, x.unsqueeze(0), scale_factor=1.0, amp=True, device=device
        )[0]
        rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = y.numpy()
        grade = sheet_grade(args.sicap_root, stem)
        cancer_frac = float((gt >= 3).mean())
        title = f"sheet {grade}  |  grade covers {cancer_frac * 100:.0f}% of the tile\n{stem}"
        path = args.out_dir / f"{grade}_{stem}.png"
        render_separate(rgb, gt, pred, title, path)
        saved.append({"stem": stem, "sheet": grade, "cancer_frac": cancer_frac, "png": str(path)})
        print(f"wrote {path}", flush=True)
    out_json = args.out_dir / "examples.json"
    out_json.write_text(json.dumps({"checkpoint": str(args.checkpoint), "examples": saved}, indent=2))
    print(f"OK {out_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-each", type=int, default=3)
    p.add_argument(
        "--stems",
        type=str,
        default="",
        help="Comma-separated tile stems. Skips the full scan and writes separate panels.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required")

    resolved = detect_arch(args.checkpoint, "auto")
    _ckpt, keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()
    print(
        f"ckpt={args.checkpoint} arch={resolved} decode_norm={decode_norm} lora={use_lora}",
        flush=True,
    )

    if args.stems.strip():
        render_named(model, device, args)
        return

    names = load_official_test_names(args.sicap_root)
    ds = SicapPatchDataset(args.sicap_root, names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rows: list[dict] = []
    nc_tot = fp_tot = ca_tot = fn_tot = 0
    for images, masks, stems in loader:
        pred = predict_batch(model, images, scale_factor=1.0, amp=True, device=device)
        gt = masks.numpy()
        for i, stem in enumerate(stems):
            st = interior_stats(pred[i], gt[i])
            st["stem"] = str(stem)
            st["slide"] = slide_id_from_patch(str(stem))
            rows.append(st)
            nc_tot += st["nc"]
            fp_tot += st["fp"]
            ca_tot += st["gt_cancer"]
            fn_tot += st["fn"]
        print(f"scanned {len(rows)}/{len(ds)}", flush=True)

    fp_rows = pick(rows, "fp", args.num_each)
    fn_rows = pick(rows, "fn", args.num_each)
    print(f"picked fp={len(fp_rows)} fn={len(fn_rows)}", flush=True)

    by_name = {name: i for i, name in enumerate(ds.names)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for kind, group in (("fp", fp_rows), ("fn", fn_rows)):
        for rank, rec in enumerate(group, start=1):
            x, y, stem = ds[by_name[rec["stem"]]]
            pred = predict_batch(
                model,
                x.unsqueeze(0),
                scale_factor=1.0,
                amp=True,
                device=device,
            )[0]
            rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt = y.numpy()
            if kind == "fp":
                title = (
                    f"Non-cancer called {CLASS_NAMES[rec['fp_grade']]}  |  {rec['stem']}\n"
                    f"{rec['fp_of_nc'] * 100:.0f}% of non-cancer pixels called cancer"
                )
            else:
                title = (
                    f"Cancer region called {CLASS_NAMES[rec['fn_class']]}  |  {rec['stem']}\n"
                    f"{rec['fn_of_cancer'] * 100:.0f}% of the cancer drawing called non-cancer"
                )
            path = args.out_dir / f"{kind}_{rank}_{rec['stem']}.png"
            render_row(rgb, gt, pred, title, path)
            rec_out = {
                **rec,
                "kind": kind,
                "png": str(path),
            }
            saved.append(rec_out)
            print(f"wrote {path}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "partition": "official",
        "n_tiles": len(rows),
        "border_px_ignored": BORDER,
        "nc_called_cancer": float(fp_tot / nc_tot) if nc_tot else None,
        "cancer_called_nc": float(fn_tot / ca_tot) if ca_tot else None,
        "examples": saved,
    }
    out_json = args.out_dir / "examples.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(
        f"official interior NC->cancer={summary['nc_called_cancer']:.3f} "
        f"cancer->NC={summary['cancer_called_nc']:.3f}",
        flush=True,
    )
    print(f"OK {out_json}", flush=True)


if __name__ == "__main__":
    main()
