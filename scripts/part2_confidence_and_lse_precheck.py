#!/usr/bin/env python3
"""Part 2 (ep21 confidence on divergent slides) + offline LSE r pre-check.

ep12 checkpoint is pruned — full differing-pixel ep12-vs-ep21 analysis is
impossible. This job:
  1) Runs ep21 on the divergent-slide sample with per-pixel max-softmax confidence
  2) Summarizes low- vs high-confidence mass (proxy for noise-padding hypothesis)
  3) On confidence-gated pixels (max_prob > conf_thr), sweeps LSE r for G5 share
     vs plain mean (mechanical effect of the formula)

Working derive_grade threshold: min_area_pct=0.05 (not chasing the monotonic edge).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
from patch_utils import NUM_CLASSES, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset, _preprocess_image  # noqa: E402


def lse_pool(values: torch.Tensor, r: float) -> float:
    """WeGleNet Eq.2 style: (1/r) log( (1/S) sum exp(r x) ) on 1D scores."""
    x = values.float().reshape(-1)
    if x.numel() == 0:
        return float("nan")
    if abs(r) < 1e-8:
        return float(x.mean().item())
    # (1/r) * (logsumexp(r*x) - log S)
    return float(((torch.logsumexp(r * x, dim=0) - math.log(x.numel())) / r).item())


class PatchDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, mode: str = "raw") -> None:
        self.base = BaselinePatchDataset(
            PROJECT / "outputs/splits/panda_train.csv",
            mode=mode,
            allow_missing_h5=True,
        )
        self.base.df = split_df.reset_index(drop=True)
        self.df = self.base.df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row["image_id"])
        rgb = self.base._read_image(sid, int(row["x"]), int(row["y"]))
        return _preprocess_image(rgb), sid


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT
        / "outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag"
        / "epoch_021_cancer_0.6356.pth",
    )
    ap.add_argument(
        "--sample-csv",
        type=Path,
        default=PROJECT / "outputs/pseudo_label/opt3_ep12_ep21_divergent_gpu_sample.csv",
    )
    ap.add_argument("--split-csv", type=Path, default=PROJECT / "outputs/splits/panda_train.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "outputs/pseudo_label/part2_lse_precheck",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--conf-thr", type=float, default=0.7)
    ap.add_argument("--min-area-pct", type=float, default=0.05)
    ap.add_argument("--max-patches-per-slide", type=int, default=64)
    ap.add_argument("--lse-r", type=float, nargs="+", default=[2, 4, 8, 16, 32])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample_csv, dtype={"slide_id": str})
    slide_ids = sample["slide_id"].astype(str).tolist()
    split = pd.read_csv(args.split_csv, dtype={"image_id": str})
    split = split[split["image_id"].isin(slide_ids)].copy()
    split = (
        split.sort_values(["image_id", "y", "x"])
        .groupby("image_id", sort=False)
        .head(args.max_patches_per_slide)
        .reset_index(drop=True)
    )
    print(
        f"slides={len(slide_ids)} patches={len(split)} "
        f"ckpt={args.checkpoint} conf_thr={args.conf_thr} min_area={args.min_area_pct}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = detect_arch(args.checkpoint, "auto")
    _, keys = _peek_state_dict(args.checkpoint)
    _, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(resolved, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(args.checkpoint, model)
    model.to(device).eval()
    print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora} device={device}")

    ds = PatchDataset(split, mode="raw")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Streaming accumulators (no full-pixel conf/prob buffers — prior 64G OOM).
    low_thr = 0.7
    bins = np.linspace(0, 1, 21)
    n_bins = len(bins) - 1
    hard_counts: dict[str, np.ndarray] = {
        s: np.zeros(NUM_CLASSES, dtype=np.int64) for s in slide_ids
    }
    # per-slide conf stats
    n_all = defaultdict(int)
    n_low_all = defaultdict(int)
    sum_all = defaultdict(float)
    hist_all_s: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_bins, dtype=np.int64))
    n_can = defaultdict(int)
    n_low_can = defaultdict(int)
    sum_can = defaultdict(float)
    hist_can_s: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_bins, dtype=np.int64))
    # LSE online: n_gate, sum_p[C], logsumexp(r*p_c) per class/r
    n_gate = defaultdict(int)
    sum_p = {s: np.zeros(NUM_CLASSES, dtype=np.float64) for s in slide_ids}
    lse_state: dict[tuple[str, float], np.ndarray] = {}  # (sid,r) -> logsumexp over pixels, shape C

    def _update_lse(sid: str, pr: np.ndarray) -> None:
        """pr: (N,C) confident pixel probs."""
        if pr.size == 0:
            return
        n_gate[sid] += int(pr.shape[0])
        sum_p[sid] += pr.sum(axis=0, dtype=np.float64)
        t = torch.from_numpy(pr.astype(np.float32, copy=False))
        for r in args.lse_r:
            key = (sid, float(r))
            chunk = torch.logsumexp(float(r) * t, dim=0).numpy()  # C
            if key not in lse_state:
                lse_state[key] = chunk.astype(np.float64)
            else:
                a = torch.from_numpy(lse_state[key])
                b = torch.from_numpy(chunk.astype(np.float64))
                lse_state[key] = torch.logaddexp(a, b).numpy()

    for step, (images, sids) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits.float(), dim=1)  # B,C,H,W
        conf, pred = probs.max(dim=1)  # B,H,W
        for i, sid in enumerate(sids):
            sid = str(sid)
            c = conf[i].reshape(-1).detach().float().cpu().numpy()
            p = pred[i].reshape(-1).detach().cpu().numpy()
            pr = (
                probs[i]
                .permute(1, 2, 0)
                .reshape(-1, NUM_CLASSES)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            n_all[sid] += int(c.size)
            n_low_all[sid] += int((c < low_thr).sum())
            sum_all[sid] += float(c.sum())
            hist_all_s[sid] += np.histogram(c, bins=bins)[0]
            cancer_m = np.isin(p, [3, 4, 5])
            if cancer_m.any():
                cc = c[cancer_m]
                n_can[sid] += int(cc.size)
                n_low_can[sid] += int((cc < low_thr).sum())
                sum_can[sid] += float(cc.sum())
                hist_can_s[sid] += np.histogram(cc, bins=bins)[0]
            gate = c > args.conf_thr
            if gate.any():
                _update_lse(sid, pr[gate])
            hard_counts[sid] += np.bincount(p, minlength=NUM_CLASSES)[:NUM_CLASSES]
            del c, p, pr
        del images, probs, conf, pred, logits
        if step % 50 == 0:
            print(f"batches {step}/{len(loader)}", flush=True)

    # --- confidence summary ---
    rows = []
    hist_all = np.zeros(n_bins, dtype=np.int64)
    hist_cancer = np.zeros(n_bins, dtype=np.int64)
    tot_n = tot_low = tot_sum = 0
    tot_cn = tot_clow = tot_csum = 0
    for sid in slide_ids:
        if n_all[sid] == 0:
            continue
        na, nc = n_all[sid], n_can[sid]
        gleason, isup = derive_grade(hard_counts[sid], min_area_pct=args.min_area_pct)
        rows.append(
            {
                "slide_id": sid,
                "n_pixels": int(na),
                "n_cancer_pixels": int(nc),
                "frac_low_conf_all": float(n_low_all[sid] / na),
                "frac_low_conf_cancer": float(n_low_can[sid] / nc) if nc else float("nan"),
                "mean_conf_all": float(sum_all[sid] / na),
                "mean_conf_cancer": float(sum_can[sid] / nc) if nc else float("nan"),
                "derived_gleason_ep21_hard": gleason,
                "derived_isup_ep21_hard": int(isup),
            }
        )
        hist_all += hist_all_s[sid]
        hist_cancer += hist_can_s[sid]
        tot_n += na
        tot_low += n_low_all[sid]
        tot_sum += sum_all[sid]
        tot_cn += nc
        tot_clow += n_low_can[sid]
        tot_csum += sum_can[sid]
    conf_df = pd.DataFrame(rows)
    conf_df.to_csv(args.out_dir / "ep21_confidence_by_slide.csv", index=False)

    conf_summary = {
        "checkpoint": str(args.checkpoint),
        "n_slides": int(len(conf_df)),
        "conf_thr_low": low_thr,
        "frac_pixels_low_conf_all": float(tot_low / tot_n) if tot_n else None,
        "frac_cancer_pixels_low_conf": float(tot_clow / tot_cn) if tot_cn else None,
        "mean_conf_all": float(tot_sum / tot_n) if tot_n else None,
        "mean_conf_cancer": float(tot_csum / tot_cn) if tot_cn else None,
        "hist_bins_left": bins[:-1].tolist(),
        "hist_all": hist_all.tolist(),
        "hist_cancer": hist_cancer.tolist(),
        "ep12_differing_pixel_analysis": "BLOCKED — ep12 ckpt pruned",
        "note": (
            "Without ep12 maps we cannot measure confidence ON DIFFERING PIXELS. "
            "This is ep21 confidence on the divergent-slide sample only — a partial proxy. "
            "Streaming accumulators (no full pixel buffers)."
        ),
    }
    (args.out_dir / "ep21_confidence_summary.json").write_text(
        json.dumps(conf_summary, indent=2), encoding="utf-8"
    )
    print("\n===== EP21 CONFIDENCE (divergent-slide sample) =====")
    print(json.dumps(conf_summary, indent=2))

    # --- LSE r pre-check from online logsumexp states ---
    lse_rows = []
    plain_g5_shares = []
    lse_g5_by_r: dict[float, list[float]] = {float(r): [] for r in args.lse_r}
    for sid in slide_ids:
        n = n_gate[sid]
        if n < 32:
            continue
        mean_p = sum_p[sid] / n
        cancer = np.clip(mean_p[3:6], 1e-12, None)
        plain_g5 = float(cancer[2] / cancer.sum())
        plain_g5_shares.append(plain_g5)
        for r in args.lse_r:
            key = (sid, float(r))
            if key not in lse_state:
                continue
            # (1/r) * (logsumexp(r x) - log N)
            lse_c = (lse_state[key] - math.log(n)) / float(r)
            cancer_l = np.clip(lse_c[3:6], 1e-12, None)
            g5 = float(cancer_l[2] / cancer_l.sum())
            lse_g5_by_r[float(r)].append(g5)
            lse_rows.append(
                {
                    "slide_id": sid,
                    "r": float(r),
                    "n_confident_pixels": int(n),
                    "plain_mean_g5_share": plain_g5,
                    "lse_g5_share": g5,
                    "delta_g5_lse_minus_plain": g5 - plain_g5,
                }
            )

    lse_df = pd.DataFrame(lse_rows)
    lse_df.to_csv(args.out_dir / "lse_r_g5_by_slide.csv", index=False)

    lse_summary = {
        "conf_gate": args.conf_thr,
        "min_area_pct_working": args.min_area_pct,
        "n_slides_with_confident_pixels": int(lse_df["slide_id"].nunique()) if len(lse_df) else 0,
        "plain_mean_g5_share_mean": float(np.mean(plain_g5_shares)) if plain_g5_shares else None,
        "by_r": {},
    }
    print("\n===== LSE r PRE-CHECK (confidence-gated pixels, G5 cancer share) =====")
    print(
        f"plain mean G5 share (mean over slides): "
        f"{lse_summary['plain_mean_g5_share_mean']}"
    )
    for r in args.lse_r:
        vals = lse_g5_by_r[float(r)]
        if not vals:
            continue
        mean_g5 = float(np.mean(vals))
        mean_delta = float(np.mean(lse_df.loc[lse_df.r == float(r), "delta_g5_lse_minus_plain"]))
        lse_summary["by_r"][str(r)] = {
            "lse_g5_share_mean": mean_g5,
            "mean_delta_vs_plain": mean_delta,
            "n_slides": len(vals),
        }
        print(f"r={r}: G5 share={mean_g5:.1%} (plain avg: {lse_summary['plain_mean_g5_share_mean']:.1%})  Δ={mean_delta:+.3f}")

    # plateau heuristic: where |ΔG5| between adjacent r drops below 0.01
    rs_sorted = sorted(float(r) for r in args.lse_r)
    plateau_note = []
    for a, b in zip(rs_sorted, rs_sorted[1:]):
        ga = lse_summary["by_r"].get(str(a), {}).get("lse_g5_share_mean")
        gb = lse_summary["by_r"].get(str(b), {}).get("lse_g5_share_mean")
        if ga is None or gb is None:
            continue
        plateau_note.append({"r_from": a, "r_to": b, "abs_delta_g5": abs(gb - ga)})
    lse_summary["adjacent_r_deltas"] = plateau_note
    (args.out_dir / "lse_r_precheck_summary.json").write_text(
        json.dumps(lse_summary, indent=2), encoding="utf-8"
    )

    # decision-gate note
    gate = {
        "part2_status": "PARTIAL — ep12 ckpt missing; cannot score differing-pixel confidence",
        "ep21_frac_cancer_pixels_low_conf": conf_summary["frac_cancer_pixels_low_conf"],
        "interpretation_hint": (
            "If ep21 cancer pixels are mostly HIGH conf on divergent slides, "
            "that leans against noise-padding as the sole driver (confident disagreement / "
            "systematic shift more likely). Full Part2 gate still needs ep12 maps."
        ),
        "confidence_gating_build": (
            "DO NOT auto-build into training until full Part2 differing-pixel test is possible "
            "OR Omar explicitly overrides the gate given this partial evidence."
        ),
        "lse_precheck": lse_summary,
        "offline_frac_shift": json.loads(
            (PROJECT / "outputs/pseudo_label/opt3_ep12_ep21_part2_offline_summary.json").read_text()
        ),
    }
    (args.out_dir / "part2_decision_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("\n===== DECISION GATE =====")
    print(json.dumps(gate, indent=2)[:2000])
    print(f"\nWrote outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
