"""Three-way ISUP-referee correction on a frozen teacher pack.

Per pixel (non-ISUP-0 slides):
  agree with expert mask                 -> keep original mask
  disagree, maxprob < tau (low conf)     -> ignore in loss (label unchanged)
  disagree, maxprob >= tau (high conf)   -> if pred is G3/G4/G5 and not in the
                                            slide's clinical {primary, secondary},
                                            swap to nearest allowed grade;
                                            else keep original mask
                                            (legal-grade fight / non-cancer)

ISUP-0 slides are skipped entirely (original label kept). Writes:
  correction_manifest.csv, skipped_slides.csv, skipped_isup0.txt,
  balance_report.json, G5_BIAS_SUMMARY.txt, <slide>_corrected.h5
Never overwrites an existing correction dir.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from patch_utils import PROJECT
from train.baseline_dataset import BaselinePatchDataset
from train.pseudo_label_rules import gleason_to_classes

CANCER = (3, 4, 5)
DEFAULT_CLINICAL = PROJECT / "data" / "train.csv"
DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"
VALIDATION_MARKER = "VALIDATION_ONLY"


def refuse_validation_only(teacher_dir: Path, allow: bool) -> None:
    marker = teacher_dir / VALIDATION_MARKER
    if marker.is_file() and not allow:
        raise SystemExit(
            f"{teacher_dir} is marked {VALIDATION_MARKER}. "
            "Refusing so this pack cannot be used for training by accident. "
            "Re-run with --allow-validation-only only for pipeline tests."
        )


def nearest_allowed(pred: int, allowed: set[int]) -> int:
    return min(allowed, key=lambda a: (abs(a - pred), a))


def load_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str})
    need = {"image_id", "isup_grade", "gleason_score"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns {missing}")
    return df.drop_duplicates("image_id")


def apply_slide(
    pred: np.ndarray,
    maxprob: np.ndarray,
    mask: np.ndarray,
    allowed: set[int],
    tau: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """pred/mask/maxprob: (N, H, W). Returns target, ignore, counts."""
    target = mask.copy()
    ignore = np.zeros(mask.shape, dtype=np.uint8)
    disagree = pred != mask
    low = disagree & (maxprob < tau)
    high = disagree & ~low
    ignore[low] = 1

    n_swap = 0
    n_swap_from = {c: 0 for c in CANCER}
    n_swap_to = {c: 0 for c in CANCER}
    if allowed:
        illegal = high & np.isin(pred, CANCER)
        for c in CANCER:
            if c in allowed:
                continue
            hit = illegal & (pred == c)
            if not hit.any():
                continue
            dest = nearest_allowed(c, allowed)
            target[hit] = dest
            n = int(hit.sum())
            n_swap += n
            n_swap_from[c] += n
            n_swap_to[dest] += n

    counts = {
        "n_pixels": int(pred.size),
        "n_agree": int((~disagree).sum()),
        "n_disagree": int(disagree.sum()),
        "n_ignore": int(ignore.sum()),
        "n_high_conf_disagree": int(high.sum()),
        "n_swap": n_swap,
        "n_swap_from_g3": n_swap_from[3],
        "n_swap_from_g4": n_swap_from[4],
        "n_swap_from_g5": n_swap_from[5],
        "n_swap_to_g3": n_swap_to[3],
        "n_swap_to_g4": n_swap_to[4],
        "n_swap_to_g5": n_swap_to[5],
    }
    return target, ignore, counts


def write_corrected(path: Path, coords: np.ndarray, target: np.ndarray, ignore: np.ndarray) -> None:
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords.astype(np.int32))
        f.create_dataset("target", data=target.astype(np.uint8), compression="gzip", compression_opts=4)
        f.create_dataset("ignore", data=ignore.astype(np.uint8), compression="gzip", compression_opts=4)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="ISUP-referee three-way correction")
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--clinical-csv", type=Path, default=None)
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument("--g5-swap-max-ratio", type=float, default=2.0)
    p.add_argument(
        "--fail-on-g5-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional block. Default off: surface G5-bias numbers only.",
    )
    p.add_argument(
        "--allow-validation-only",
        action="store_true",
        help="Required to run against a pack directory that contains VALIDATION_ONLY.",
    )
    p.add_argument("--max-slides", type=int, default=None)
    args = p.parse_args()

    refuse_validation_only(args.teacher_dir, args.allow_validation_only)

    if args.out_dir.exists() and any(args.out_dir.glob("*")):
        raise SystemExit(f"Refusing to overwrite existing correction dir {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pack_cfg = {}
    cfg_path = args.teacher_dir / "pack_config.json"
    if cfg_path.is_file():
        pack_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    clinical_path = args.clinical_csv or (args.teacher_dir / "clinical_isup.csv")
    if not clinical_path.is_file():
        clinical_path = DEFAULT_CLINICAL
    clinical = load_clinical(clinical_path)
    clin = clinical.set_index("image_id")

    split = pd.read_csv(args.split, dtype={"image_id": str})
    slide_ids = sorted(split["image_id"].astype(str).unique())
    if args.max_slides:
        slide_ids = slide_ids[: args.max_slides]

    ds = BaselinePatchDataset(args.split, mode="raw", allow_missing_h5=True)
    rows = []
    totals = {k: 0 for k in (
        "n_pixels", "n_agree", "n_disagree", "n_ignore", "n_high_conf_disagree",
        "n_swap", "n_swap_from_g3", "n_swap_from_g4", "n_swap_from_g5",
        "n_swap_to_g3", "n_swap_to_g4", "n_swap_to_g5", "n_isup0_skipped",
        "n_missing_pack", "n_written",
    )}
    pred_cancer = {3: 0, 4: 0, 5: 0}
    mask_pixels = 0
    mask_g5_pixels = 0

    for i, slide_id in enumerate(slide_ids, start=1):
        h5_path = args.teacher_dir / f"{slide_id}_srcpred.h5"
        rec = {"slide_id": slide_id, "skipped": "", "rule": ""}
        if slide_id not in clin.index:
            rec["skipped"] = "no_clinical"
            rows.append(rec)
            continue
        gleason = str(clin.loc[slide_id, "gleason_score"])
        isup = int(clin.loc[slide_id, "isup_grade"])
        rec["metadata_gleason"] = gleason
        rec["metadata_isup"] = isup
        if isup == 0:
            rec["skipped"] = "isup0"
            totals["n_isup0_skipped"] += 1
            rows.append(rec)
            continue
        if not h5_path.is_file():
            rec["skipped"] = "missing_teacher_h5"
            totals["n_missing_pack"] += 1
            rows.append(rec)
            continue
        allowed = gleason_to_classes(gleason)
        rec["allowed"] = "|".join(str(c) for c in sorted(allowed))
        with h5py.File(h5_path, "r") as f:
            coords = np.asarray(f["coords"])
            pred = np.asarray(f["preds"])
            if "maxprob" not in f:
                raise SystemExit(f"{h5_path} has no maxprob — re-run teacher cache with --write-maxprob")
            maxprob = np.asarray(f["maxprob"], dtype=np.float32)

        masks = []
        for x, y in coords:
            masks.append(ds._read_mask(slide_id, int(x), int(y)))
        mask = np.stack(masks, axis=0)
        if mask.shape != pred.shape:
            raise RuntimeError(f"{slide_id}: mask {mask.shape} != pred {pred.shape}")

        for c in CANCER:
            pred_cancer[c] += int((pred == c).sum())
        mask_pixels += int(mask.size)
        mask_g5_pixels += int((mask == 5).sum())

        target, ignore, counts = apply_slide(pred, maxprob, mask, allowed, args.conf_threshold)
        rec.update(counts)
        rec["rule"] = "three_way_isup_referee"
        for k, v in counts.items():
            totals[k] += v
        if counts["n_swap"] or counts["n_ignore"]:
            write_corrected(args.out_dir / f"{slide_id}_corrected.h5", coords, target, ignore)
            totals["n_written"] += 1
            rec["corrected_h5"] = f"{slide_id}_corrected.h5"
        else:
            rec["corrected_h5"] = ""
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(slide_ids)} slides  swaps={totals['n_swap']} ignore={totals['n_ignore']}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out_dir / "correction_manifest.csv", index=False)
    skipped = manifest[manifest["skipped"].fillna("") != ""][["slide_id", "skipped"]]
    skipped.to_csv(args.out_dir / "skipped_slides.csv", index=False)
    isup0 = skipped[skipped["skipped"] == "isup0"]
    (args.out_dir / "skipped_isup0.txt").write_text(
        "ISUP-0 slides skipped (original label kept, no referee):\n"
        + "\n".join(isup0["slide_id"].astype(str).tolist())
        + ("\n" if len(isup0) else ""),
        encoding="utf-8",
    )
    print(f"ISUP-0 skipped: {int(totals['n_isup0_skipped'])} slides (see skipped_isup0.txt)", flush=True)

    n_swap = totals["n_swap"]
    swap_to_g5_share = (totals["n_swap_to_g5"] / n_swap) if n_swap else 0.0
    mask_g5_share = (mask_g5_pixels / mask_pixels) if mask_pixels else 0.0
    g5_from_share = (totals["n_swap_from_g5"] / n_swap) if n_swap else 0.0
    cancer_pred = max(sum(pred_cancer.values()), 1)
    g5_pred_share = pred_cancer[5] / cancer_pred
    limit = max(0.05, args.g5_swap_max_ratio * g5_pred_share)
    g5_bias = n_swap > 0 and g5_from_share > limit

    g5_summary = {
        "n_high_conf_swap": n_swap,
        "n_swap_to_g5": totals["n_swap_to_g5"],
        "pct_high_conf_swaps_to_g5": 100.0 * swap_to_g5_share,
        "n_original_mask_pixels": mask_pixels,
        "n_original_mask_g5": mask_g5_pixels,
        "pct_original_mask_g5": 100.0 * mask_g5_share,
        "note": "surface only — no blocking gate yet",
    }
    print(
        "G5-bias summary (no gate): "
        f"{g5_summary['pct_high_conf_swaps_to_g5']:.2f}% of high-conf swaps "
        f"resulted in G5 ({totals['n_swap_to_g5']}/{n_swap}); "
        f"original mask G5 share {g5_summary['pct_original_mask_g5']:.2f}% "
        f"({mask_g5_pixels}/{mask_pixels} pixels).",
        flush=True,
    )

    report = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "teacher_dir": str(args.teacher_dir.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "conf_threshold": args.conf_threshold,
        "allow_validation_only": bool(args.allow_validation_only),
        "pack_config": pack_cfg,
        "n_slides": len(slide_ids),
        "totals": totals,
        "teacher_cancer_pred_share": {str(k): v / cancer_pred for k, v in pred_cancer.items()},
        "swap_from_g5_share": g5_from_share,
        "swap_to_g5_share": swap_to_g5_share,
        "g5_pred_share": g5_pred_share,
        "original_mask_g5_share": mask_g5_share,
        "g5_summary": g5_summary,
        "g5_swap_limit": limit,
        "g5_bias_flag": g5_bias,
        "gate": "FAIL_G5_BIAS" if (g5_bias and args.fail_on_g5_bias) else "PASS",
    }
    (args.out_dir / "balance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "G5_BIAS_SUMMARY.txt").write_text(
        (
            f"pct_high_conf_swaps_to_g5={g5_summary['pct_high_conf_swaps_to_g5']:.4f}\n"
            f"n_swap_to_g5={totals['n_swap_to_g5']}\n"
            f"n_high_conf_swap={n_swap}\n"
            f"pct_original_mask_g5={g5_summary['pct_original_mask_g5']:.4f}\n"
            f"n_original_mask_g5={mask_g5_pixels}\n"
            f"n_original_mask_pixels={mask_pixels}\n"
            "gate=surface_only\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("gate", "g5_summary", "totals")}, indent=2))
    if g5_bias and args.fail_on_g5_bias:
        raise SystemExit(
            f"G5 bias gate: {g5_from_share:.3f} of high-conf swaps came from G5 "
            f"(teacher G5 pred share {g5_pred_share:.3f}, limit {limit:.3f}). "
            "Do not start Round N+1. See balance_report.json."
        )


if __name__ == "__main__":
    main()
