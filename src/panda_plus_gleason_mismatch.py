"""PANDA+: mask-derived vs model-derived Gleason (primary/secondary) mismatch.

Aggregates G3/G4/G5 pixel counts per slide from PANDA+ patch masks and from
model argmax predictions, then derives Gleason with the same min_area_pct gate
used in the train ISUP diagnostic (default 5%).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from isup_diagnostic import derive_grade  # noqa: E402
from patch_utils import NUM_CLASSES, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402

DEFAULT_SPLIT = PROJECT / "outputs" / "panda_plus" / "panda_plus_patches.csv"
DEFAULT_MASK_DIR = PROJECT / "outputs" / "panda_plus" / "masks"
DEFAULT_CKPT = (
    PROJECT
    / "outputs"
    / "checkpoints"
    / "uni2_upernet_raw_h200x4"
    / "epoch_042_cancer_0.7420.pth"
)
DEFAULT_OUT = PROJECT / "outputs" / "pseudo_label" / "panda_plus_gleason_mismatch.csv"


def parse_gleason(g: str) -> tuple[int | None, int | None]:
    if g in {"benign", "negative", "0+0", ""}:
        return None, None
    a, b = g.split("+")
    return int(a), int(b)


def summarize(rows: pd.DataFrame) -> dict:
    n = len(rows)
    prim_ok = int((rows["mask_primary"] == rows["model_primary"]).sum())
    sec_ok = int((rows["mask_secondary"] == rows["model_secondary"]).sum())
    both_ok = int(
        (
            (rows["mask_primary"] == rows["model_primary"])
            & (rows["mask_secondary"] == rows["model_secondary"])
        ).sum()
    )
    pattern_ok = int((rows["mask_gleason"] == rows["model_gleason"]).sum())
    isup_ok = int((rows["mask_isup"] == rows["model_isup"]).sum())

    def rate(k: int) -> float:
        return float(k / n) if n else 0.0

    conf_p = (
        pd.crosstab(rows["mask_primary"], rows["model_primary"], dropna=False)
        .reindex(index=[3, 4, 5], columns=[3, 4, 5], fill_value=0)
    )
    conf_s = (
        pd.crosstab(rows["mask_secondary"], rows["model_secondary"], dropna=False)
        .reindex(index=[3, 4, 5], columns=[3, 4, 5], fill_value=0)
    )
    top = (
        rows.loc[rows["mask_gleason"] != rows["model_gleason"], ["mask_gleason", "model_gleason"]]
        .value_counts()
        .head(15)
    )
    return {
        "n_slides": n,
        "primary_match": prim_ok,
        "primary_match_rate": rate(prim_ok),
        "primary_mismatch_rate": rate(n - prim_ok),
        "secondary_match": sec_ok,
        "secondary_match_rate": rate(sec_ok),
        "secondary_mismatch_rate": rate(n - sec_ok),
        "primary_and_secondary_match": both_ok,
        "primary_and_secondary_match_rate": rate(both_ok),
        "pattern_match": pattern_ok,
        "pattern_match_rate": rate(pattern_ok),
        "pattern_mismatch_rate": rate(n - pattern_ok),
        "isup_match": isup_ok,
        "isup_match_rate": rate(isup_ok),
        "isup_mismatch_rate": rate(n - isup_ok),
        "primary_confusion_mask_rows_model_cols": conf_p.to_dict(),
        "secondary_confusion_mask_rows_model_cols": conf_s.to_dict(),
        "top_pattern_mismatches": [
            {"mask": str(a), "model": str(b), "n": int(c)} for (a, b), c in top.items()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    ap.add_argument("--mask-suffix", default="_pandaplus_mask.png")
    ap.add_argument("--min-area-pct", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--pred-on-labeled-only",
        action="store_true",
        help="Count model cancer pixels only where mask is labeled (gt>=2).",
    )
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    if os.environ.get("TORCH_CUDNN_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        torch.backends.cudnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BaselinePatchDataset(
        args.split,
        mode="raw",
        allow_missing_h5=True,
        mask_dir=args.mask_dir,
        mask_suffix=args.mask_suffix,
        prefer_h5_masks=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    arch = detect_arch(args.checkpoint, "uni2_upernet")
    _ckpt_peek, peek_keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(peek_keys)
    model = build_eval_model(arch, decode_norm=decode_norm, use_lora=use_lora)
    ckpt = load_model_weights(args.checkpoint, model)
    if _is_opt3 or use_lora:
        print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora}", flush=True)
    model.to(device)
    model.eval()

    slide_ids = [str(x) for x in ds.df["image_id"].tolist()]
    unique_slides = sorted(set(slide_ids))
    slide_to_i = {s: i for i, s in enumerate(unique_slides)}
    mask_counts = np.zeros((len(unique_slides), NUM_CLASSES), dtype=np.int64)
    pred_counts = np.zeros((len(unique_slides), NUM_CLASSES), dtype=np.int64)
    n_patches = np.zeros(len(unique_slides), dtype=np.int64)

    print(
        f"PANDA+ Gleason mismatch: slides={len(unique_slides)} patches={len(ds)} "
        f"device={device} ckpt={args.checkpoint}",
        flush=True,
    )

    idx = 0
    with torch.inference_mode():
        for images, masks, _w in loader:
            b = images.size(0)
            batch_slides = slide_ids[idx : idx + b]
            idx += b
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                preds = model(images).argmax(dim=1)

            for j, sid in enumerate(batch_slides):
                si = slide_to_i[sid]
                n_patches[si] += 1
                m = masks[j].reshape(-1)
                p = preds[j].reshape(-1)
                mask_counts[si] += torch.bincount(m, minlength=NUM_CLASSES).cpu().numpy()
                if args.pred_on_labeled_only:
                    labeled = m >= 2
                    if labeled.any():
                        pred_counts[si] += torch.bincount(
                            p[labeled], minlength=NUM_CLASSES
                        ).cpu().numpy()
                else:
                    pred_counts[si] += torch.bincount(p, minlength=NUM_CLASSES).cpu().numpy()

    rows = []
    for sid in unique_slides:
        si = slide_to_i[sid]
        m_g, m_isup = derive_grade(mask_counts[si], min_area_pct=args.min_area_pct)
        p_g, p_isup = derive_grade(pred_counts[si], min_area_pct=args.min_area_pct)
        mp, ms = parse_gleason(m_g)
        pp, ps = parse_gleason(p_g)
        mc = mask_counts[si]
        pc = pred_counts[si]
        m_cancer = int(mc[3:6].sum())
        p_cancer = int(pc[3:6].sum())
        rows.append(
            {
                "slide_id": sid,
                "n_patches": int(n_patches[si]),
                "mask_gleason": m_g,
                "mask_isup": int(m_isup),
                "mask_primary": mp,
                "mask_secondary": ms,
                "model_gleason": p_g,
                "model_isup": int(p_isup),
                "model_primary": pp,
                "model_secondary": ps,
                "pattern_match": m_g == p_g,
                "primary_match": mp == pp,
                "secondary_match": ms == ps,
                "isup_match": int(m_isup) == int(p_isup),
                **{f"mask_pixels_{c}": int(mc[c]) for c in range(NUM_CLASSES)},
                **{f"pred_pixels_{c}": int(pc[c]) for c in range(NUM_CLASSES)},
                **{
                    f"mask_cancer_frac_g{g}": (float(mc[g] / m_cancer) if m_cancer else 0.0)
                    for g in (3, 4, 5)
                },
                **{
                    f"model_cancer_frac_g{g}": (float(pc[g] / p_cancer) if p_cancer else 0.0)
                    for g in (3, 4, 5)
                },
            }
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "split": str(args.split),
            "min_area_pct": float(args.min_area_pct),
            "pred_on_labeled_only": bool(args.pred_on_labeled_only),
            "n_patches": int(len(ds)),
        }
    )
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== PANDA+ mask vs model Gleason ===")
    print(f"slides: {summary['n_slides']}")
    print(
        f"primary   match: {summary['primary_match']}/{summary['n_slides']} "
        f"= {summary['primary_match_rate']:.1%}  "
        f"(mismatch {summary['primary_mismatch_rate']:.1%})"
    )
    print(
        f"secondary match: {summary['secondary_match']}/{summary['n_slides']} "
        f"= {summary['secondary_match_rate']:.1%}  "
        f"(mismatch {summary['secondary_mismatch_rate']:.1%})"
    )
    print(
        f"both (1st+2nd):  {summary['primary_and_secondary_match']}/{summary['n_slides']} "
        f"= {summary['primary_and_secondary_match_rate']:.1%}"
    )
    print(
        f"full pattern:    {summary['pattern_match']}/{summary['n_slides']} "
        f"= {summary['pattern_match_rate']:.1%}  "
        f"(mismatch {summary['pattern_mismatch_rate']:.1%})"
    )
    print(
        f"ISUP:            {summary['isup_match']}/{summary['n_slides']} "
        f"= {summary['isup_match_rate']:.1%}  "
        f"(mismatch {summary['isup_mismatch_rate']:.1%})"
    )
    print("\nTop pattern mismatches (mask→model):")
    for item in summary["top_pattern_mismatches"]:
        print(f"  {item['mask']} → {item['model']}: {item['n']}")
    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
