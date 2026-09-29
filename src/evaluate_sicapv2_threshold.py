"""SICAPv2 inference-side cancer-probability threshold grid search.

Fair protocol (no weight updates, no Test leakage):
  For each Val fold: sweep thresholds on Train.xlsx → pick t* that max
  binary cancer Dice → apply t* on Test.xlsx.

Decision rule (multi-class → SICAP labels):
  p_cancer = softmax(logits)[G3]+[G4]+[G5]
  pred = remap(argmax(logits))   # bg/stroma/benign → NC
  pred[p_cancer < t] = NC        # gate low-confidence cancer calls

Our main failure mode is NC→cancer FPs, so optimal t is often >0.5;
we still sweep low thresholds as in the binary-seg literature.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from evaluate_sicapv2 import (  # noqa: E402
    DEFAULT_ROOT,
    SICAP_CLASSES,
    SicapPatchDataset,
    dice_from_cm,
    load_split_names,
    predict_batch_logits,
    remap_panda_pred,
)
from patch_utils import OUTPUTS  # noqa: E402


def accumulate_threshold_cms(
    model: torch.nn.Module,
    device: torch.device,
    root: Path,
    names: list[str],
    thresholds: list[float],
    *,
    scale_factor: float,
    batch_size: int,
    num_workers: int,
    amp: bool,
    max_patches: int | None,
) -> tuple[dict[float, np.ndarray], int, int]:
    if max_patches is not None and max_patches > 0 and len(names) > max_patches:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(names), size=max_patches, replace=False)
        names = [names[i] for i in sorted(idx.tolist())]
        print(f"  subsampled to {len(names)} patches for this split", flush=True)

    ds = SicapPatchDataset(root, names)
    bs = 1 if scale_factor > 1.0 else batch_size
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    cms = {t: np.zeros((6, 6), dtype=np.int64) for t in thresholds}
    n_pix = 0
    for images, masks, _ in loader:
        logits = predict_batch_logits(
            model, images, scale_factor=scale_factor, amp=amp, device=device
        )
        probs = F.softmax(logits.float(), dim=1)
        p_cancer = (probs[:, 3] + probs[:, 4] + probs[:, 5]).cpu().numpy()
        base = remap_panda_pred(logits.argmax(dim=1).cpu().numpy())
        gt = np.clip(masks.numpy(), 0, 5).astype(np.int64)
        for t in thresholds:
            pred = base.copy()
            pred[p_cancer < t] = 0
            pred = np.clip(pred, 0, 5).astype(np.int64)
            flat_gt = gt.ravel()
            flat_pr = pred.ravel()
            cms[t] += np.bincount(flat_gt * 6 + flat_pr, minlength=36).reshape(6, 6)
        n_pix += int(gt.size)
    return cms, len(ds), ds.skipped


def main() -> None:
    p = argparse.ArgumentParser(description="SICAPv2 threshold grid search")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--folds", type=str, default="1,2,3,4")
    p.add_argument("--scale-factor", type=float, default=2.0)
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85",
    )
    p.add_argument(
        "--tune-metric",
        choices=("binary_cancer", "cancer"),
        default="binary_cancer",
        help="Metric maximized on Train to pick t*",
    )
    p.add_argument(
        "--tune-max-patches",
        type=int,
        default=2500,
        help="Cap Train patches per fold for tuning speed (0=all)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    root = args.sicap_root
    tune_max = None if args.tune_max_patches <= 0 else args.tune_max_patches

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
        f"ckpt={args.checkpoint} scale={args.scale_factor} "
        f"thresholds={thresholds} tune_metric={args.tune_metric}",
        flush=True,
    )

    fold_rows = []
    for fold in folds:
        print(f"=== Val{fold} tune on Train ===", flush=True)
        train_names = load_split_names(root, fold, "Train")
        train_cms, n_tr, skip_tr = accumulate_threshold_cms(
            model,
            device,
            root,
            train_names,
            thresholds,
            scale_factor=args.scale_factor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            max_patches=tune_max,
        )
        train_curves = []
        best_t = thresholds[0]
        best_val = -1.0
        for t in thresholds:
            m = dice_from_cm(train_cms[t], SICAP_CLASSES)
            train_curves.append({"threshold": t, **{k: m[k] for k in ("NC", "G3", "G4", "G5", "mean4", "cancer", "binary_cancer")}})
            score = float(m[args.tune_metric])
            print(
                f"  train t={t:.2f} cancer={m['cancer']:.4f} bin={m['binary_cancer']:.4f}",
                flush=True,
            )
            if score > best_val:
                best_val = score
                best_t = t
        print(f"  → t*={best_t:.2f} (train {args.tune_metric}={best_val:.4f})", flush=True)

        print(f"=== Val{fold} Test @ t*={best_t:.2f} ===", flush=True)
        test_names = load_split_names(root, fold, "Test")
        test_cms, n_te, skip_te = accumulate_threshold_cms(
            model,
            device,
            root,
            test_names,
            thresholds,  # full curve on test for analysis; headline uses t*
            scale_factor=args.scale_factor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            max_patches=None,
        )
        test_curves = []
        for t in thresholds:
            m = dice_from_cm(test_cms[t], SICAP_CLASSES)
            test_curves.append({"threshold": t, **{k: m[k] for k in ("NC", "G3", "G4", "G5", "mean4", "cancer", "binary_cancer")}})
        m_star = dice_from_cm(test_cms[best_t], SICAP_CLASSES)
        # also argmax baseline t≈0 (no gate) approximated by very low t if in list, else report t=0.0 separately
        print(
            f"  test@t* cancer={m_star['cancer']:.4f} mean4={m_star['mean4']:.4f} "
            f"bin={m_star['binary_cancer']:.4f}  "
            f"NC={m_star['NC']:.4f} G3={m_star['G3']:.4f} G4={m_star['G4']:.4f} G5={m_star['G5']:.4f}",
            flush=True,
        )
        fold_rows.append(
            {
                "fold": fold,
                "t_star": best_t,
                "train_tune_score": best_val,
                "n_train_scored": n_tr,
                "n_test_scored": n_te,
                "n_train_skipped": skip_tr,
                "n_test_skipped": skip_te,
                **{k: m_star[k] for k in ("NC", "G3", "G4", "G5", "mean4", "cancer", "binary_cancer")},
                "train_curve": train_curves,
                "test_curve": test_curves,
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sicap_root": str(root.resolve()),
        "scale_factor": float(args.scale_factor),
        "thresholds": thresholds,
        "tune_metric": args.tune_metric,
        "tune_max_patches": tune_max,
        "protocol": (
            "fair_train_tune_test_apply; "
            "p_cancer=softmax G3+G4+G5; gate pred→NC if p_cancer<t; "
            "remap bg/stroma/benign→NC"
        ),
        "folds": folds,
        "per_fold": fold_rows,
        "mean_cancer_dice": float(np.mean([r["cancer"] for r in fold_rows])),
        "mean_mean4_dice": float(np.mean([r["mean4"] for r in fold_rows])),
        "mean_binary_cancer_dice": float(np.mean([r["binary_cancer"] for r in fold_rows])),
        "mean_t_star": float(np.mean([r["t_star"] for r in fold_rows])),
    }
    print(
        f"MEAN test@t* cancer={summary['mean_cancer_dice']:.4f} "
        f"mean4={summary['mean_mean4_dice']:.4f} "
        f"binary={summary['mean_binary_cancer_dice']:.4f} "
        f"mean_t*={summary['mean_t_star']:.3f}",
        flush=True,
    )

    out_dir = args.out_dir or (
        OUTPUTS
        / "evaluation"
        / "sicapv2"
        / f"thresh_mpp{args.scale_factor:g}x_{args.checkpoint.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sicapv2_threshold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "sicapv2_threshold_per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold", "t_star", "train_tune_score",
                "NC", "G3", "G4", "G5", "mean4", "cancer", "binary_cancer",
            ],
        )
        w.writeheader()
        for r in fold_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
