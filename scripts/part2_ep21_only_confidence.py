#!/usr/bin/env python3
"""Modified Part 2 — single-checkpoint (ep21) confidence diagnosis.

LIMITATION (explicit): ep12 is unrecoverable, so this is NOT the originally
planned two-checkpoint differing-pixel comparison. It answers a weaker but
usable version of the same question with only ep21:

  A) On slides where ep21 derived ISUP ≠ clinical: confidence of pixels that
     drive the ISUP aggregate (predicted G3/G4/G5).
  B) Where ep21 hard pred ≠ expert mask: confidence of those disagreeing pixels.

Working min_area_pct = 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    detect_arch,
    load_model_weights,
)
from isup_diagnostic import derive_grade  # noqa: E402
from patch_utils import MASKS_DIR, NUM_CLASSES, PATCH_SIZE, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset, _preprocess_image  # noqa: E402


class PatchMaskDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, mode: str = "raw") -> None:
        self.base = BaselinePatchDataset(
            PROJECT / "outputs/splits/panda_train.csv",
            mode=mode,
            allow_missing_h5=True,
            mask_dir=MASKS_DIR,
            mask_suffix="_mask.tiff",
            prefer_h5_masks=False,
        )
        self.base.df = split_df.reset_index(drop=True)
        self.df = self.base.df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])
        rgb = self.base._read_image(sid, x, y)
        # Expert mask patch (TIFF via BaselinePatchDataset / openslide)
        try:
            mask = self.base._read_mask(sid, x, y)
        except Exception:
            import openslide

            msl = openslide.OpenSlide(str(MASKS_DIR / f"{sid}_mask.tiff"))
            raw = np.array(msl.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)))
            msl.close()
            mask = raw[:, :, 0] if raw.ndim == 3 else raw
        mask = np.clip(np.asarray(mask), 0, 5).astype(np.int64)
        return _preprocess_image(rgb), torch.from_numpy(mask), sid


def _hist(conf: np.ndarray, bins: np.ndarray) -> list[int]:
    if conf.size == 0:
        return [0] * (len(bins) - 1)
    h, _ = np.histogram(conf, bins=bins)
    return h.astype(int).tolist()


def _quantile_from_hist(counts: np.ndarray, bins: np.ndarray, q: float) -> float:
    """Approximate quantile from a histogram (bin midpoints)."""
    total = int(counts.sum())
    if total <= 0:
        return float("nan")
    target = q * total
    csum = 0
    for i, c in enumerate(counts):
        csum += int(c)
        if csum >= target:
            return float(0.5 * (bins[i] + bins[i + 1]))
    return float(bins[-1])


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT
        / "outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag"
        / "epoch_021_cancer_0.6356.pth",
    )
    ap.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=PROJECT / "outputs/pseudo_label/diagnostic_report_opt3_ep21.csv",
    )
    ap.add_argument("--split-csv", type=Path, default=PROJECT / "outputs/splits/panda_train.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "outputs/pseudo_label/part2_ep21_only",
    )
    ap.add_argument("--conf-thr", type=float, default=0.7)
    ap.add_argument("--min-area-pct", type=float, default=0.05)
    ap.add_argument("--max-mismatch-slides", type=int, default=400)
    ap.add_argument("--max-match-control-slides", type=int, default=80)
    ap.add_argument("--max-patches-per-slide", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    diag = pd.read_csv(args.diagnostic_csv, dtype={"slide_id": str})
    mismatch = diag[~diag["match"]].copy()
    match_ok = diag[diag["match"]].copy()
    # stratified sample mismatch by clinical ISUP
    parts = []
    for g, grp in mismatch.groupby("metadata_isup"):
        n = min(len(grp), max(20, args.max_mismatch_slides // 6))
        parts.append(grp.sample(n=n, random_state=0) if len(grp) > n else grp)
    mismatch_s = pd.concat(parts).drop_duplicates("slide_id")
    if len(mismatch_s) > args.max_mismatch_slides:
        mismatch_s = mismatch_s.sample(n=args.max_mismatch_slides, random_state=1)
    control = match_ok.sample(
        n=min(args.max_match_control_slides, len(match_ok)), random_state=2
    )
    chosen = pd.concat([mismatch_s, control]).drop_duplicates("slide_id")
    mismatch_ids = set(mismatch_s["slide_id"].astype(str))
    chosen.to_csv(args.out_dir / "slide_sample.csv", index=False)
    print(
        f"LIMITATION: single-checkpoint (ep21) analysis — NOT ep12-vs-ep21 differing pixels.\n"
        f"mismatch_slides={len(mismatch_s)} control_match_slides={len(control)} "
        f"conf_thr={args.conf_thr} min_area_pct={args.min_area_pct}"
    )

    split = pd.read_csv(args.split_csv, dtype={"image_id": str})
    split = split[split["image_id"].isin(chosen["slide_id"].astype(str))].copy()
    split = (
        split.sort_values(["image_id", "y", "x"])
        .groupby("image_id", sort=False)
        .head(args.max_patches_per_slide)
        .reset_index(drop=True)
    )
    print(f"patches={len(split)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = detect_arch(args.checkpoint, "auto")
    _, keys = _peek_state_dict(args.checkpoint)
    _, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()
    print(f"load decode_norm={decode_norm} use_lora={use_lora} device={device}")

    ds = PatchMaskDataset(split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Streaming conf stats (hist + sum/count) — avoid concatenating all pixels (64G OOM).
    thr = args.conf_thr
    bins = np.linspace(0, 1, 21)
    n_bins = len(bins) - 1

    def _empty_acc() -> dict:
        return {
            "n": 0,
            "n_low": 0,
            "sum": 0.0,
            "hist": np.zeros(n_bins, dtype=np.int64),
        }

    acc_drive = _empty_acc()
    acc_dis = _empty_acc()
    acc_agr = _empty_acc()
    acc_dis_c = _empty_acc()

    def _upd(acc: dict, c: np.ndarray) -> None:
        if c.size == 0:
            return
        acc["n"] += int(c.size)
        acc["n_low"] += int((c < thr).sum())
        acc["sum"] += float(c.sum())
        acc["hist"] += np.histogram(c, bins=bins)[0]

    # sanity: recompute derived ISUP from streamed hard counts on mismatch sample
    hard = {s: np.zeros(NUM_CLASSES, np.int64) for s in mismatch_ids}

    for step, (images, masks, sids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        masks_np = masks.numpy()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits.float(), dim=1)
        conf, pred = probs.max(dim=1)
        conf_np = conf.detach().float().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for i, sid in enumerate(sids):
            sid = str(sid)
            c = conf_np[i].reshape(-1)
            p = pred_np[i].reshape(-1)
            m = np.clip(masks_np[i].reshape(-1), 0, 5).astype(np.int64)

            # B) pred vs mask
            valid = (m >= 0) & (m <= 5)
            dis = valid & (p != m)
            agr = valid & (p == m)
            if dis.any():
                _upd(acc_dis, c[dis])
                cancer_dis = dis & np.isin(p, [3, 4, 5])
                if cancer_dis.any():
                    _upd(acc_dis_c, c[cancer_dis])
            if agr.any():
                _upd(acc_agr, c[agr])

            # A) ISUP-mismatch driving pixels = predicted cancer on mismatch slides
            if sid in mismatch_ids:
                drive = np.isin(p, [3, 4, 5])
                if drive.any():
                    _upd(acc_drive, c[drive])
                hard[sid] += np.bincount(p, minlength=NUM_CLASSES)[:NUM_CLASSES]

        del images, masks, probs, conf, pred, logits, conf_np, pred_np
        if step % 40 == 0:
            print(f"batches {step}/{len(loader)}", flush=True)

    def breakdown(name: str, acc: dict) -> dict:
        n = int(acc["n"])
        if n == 0:
            return {"name": name, "n_pixels": 0}
        frac_low = float(acc["n_low"] / n)
        return {
            "name": name,
            "n_pixels": n,
            "frac_low_conf_lt_0.7": frac_low,
            "frac_high_conf_ge_0.7": float(1.0 - frac_low),
            "mean_conf": float(acc["sum"] / n),
            "median_conf": _quantile_from_hist(acc["hist"], bins, 0.50),
            "p10_conf": _quantile_from_hist(acc["hist"], bins, 0.10),
            "p90_conf": _quantile_from_hist(acc["hist"], bins, 0.90),
            "hist_bins_left": bins[:-1].tolist(),
            "hist_counts": acc["hist"].astype(int).tolist(),
            "quantiles": "approx_from_hist",
        }

    # per clinical ISUP on mismatch slides: need clinical map
    clin = mismatch_s.set_index("slide_id")["metadata_isup"].to_dict()
    # Re-run light: we didn't store per-isup; compute from slide-level file after
    # Optional: derive from diagnostic for mismatch sample only using streamed hard
    slide_rows = []
    for sid in mismatch_ids:
        g, isup = derive_grade(hard[sid], min_area_pct=args.min_area_pct)
        slide_rows.append(
            {
                "slide_id": sid,
                "clinical_isup": int(clin.get(sid, -1)),
                "derived_isup_streamed": int(isup),
                "derived_gleason_streamed": g,
            }
        )
    pd.DataFrame(slide_rows).to_csv(args.out_dir / "mismatch_slides_streamed_isup.csv", index=False)

    report = {
        "analysis_type": "single_checkpoint_ep21_only",
        "limitation": (
            "ep12 checkpoint unrecoverable. This is NOT the originally planned "
            "two-checkpoint differing-pixel comparison (ep12 vs ep21). Results are a "
            "weaker but usable proxy for whether mismatches are low-confidence noise "
            "vs high-confidence wrongness."
        ),
        "checkpoint": str(args.checkpoint),
        "min_area_pct": args.min_area_pct,
        "conf_threshold": thr,
        "n_mismatch_slides_analyzed": int(len(mismatch_s)),
        "n_control_match_slides": int(len(control)),
        "n_patches": int(len(split)),
        "A_isup_mismatch_driving_pixels": {
            "definition": (
                "On slides where ep21 derived ISUP ≠ clinical (diagnostic CSV), "
                "pixels with predicted class in {G3,G4,G5} — these drive derive_grade."
            ),
            **breakdown("isup_mismatch_cancer_preds", acc_drive),
        },
        "B_pred_vs_expert_mask_disagree": {
            "definition": (
                "Any pixel where hard pred ≠ expert mask label (mask classes 0–5), "
                "pooled over mismatch+control sample slides."
            ),
            **breakdown("pred_ne_mask", acc_dis),
            "agree_pixels_for_contrast": breakdown("pred_eq_mask", acc_agr),
            "disagree_and_pred_cancer": breakdown("pred_ne_mask_and_pred_G345", acc_dis_c),
        },
        "decision_hints": {
            "if_A_mostly_low_conf": "Supports noise-padding / low-confidence aggregate pollution → confidence-gating more relevant",
            "if_A_mostly_high_conf": "Supports confident-wrong systematic bias → Rules/label correction more relevant than gating alone",
            "if_B_mostly_high_conf": "Model is confidently disagreeing with expert masks (bias), not just uncertain boundaries",
        },
    }

    # dominant mode flags (explicit, not a lock-in)
    a_low = report["A_isup_mismatch_driving_pixels"].get("frac_low_conf_lt_0.7")
    b_low = report["B_pred_vs_expert_mask_disagree"].get("frac_low_conf_lt_0.7")
    report["summary_flags"] = {
        "A_low_conf_dominates_gt50pct": bool(a_low is not None and a_low > 0.5),
        "B_low_conf_dominates_gt50pct": bool(b_low is not None and b_low > 0.5),
        "A_frac_low_conf": a_low,
        "B_frac_low_conf": b_low,
    }

    out_json = args.out_dir / "part2_ep21_only_summary.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n===== A) ISUP-mismatch driving pixels (pred G3/G4/G5) =====")
    print(json.dumps(report["A_isup_mismatch_driving_pixels"], indent=2))
    print("\n===== B) pred ≠ expert mask =====")
    print(json.dumps(report["B_pred_vs_expert_mask_disagree"], indent=2))
    print("\n===== FLAGS =====")
    print(json.dumps(report["summary_flags"], indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
