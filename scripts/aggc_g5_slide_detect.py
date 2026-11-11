#!/usr/bin/env python3
"""Slide-level G5 detector from per-slide AGGC CSVs.

Score = predicted G5 pixels on annotated tissue / annotated pixels.
Reports pixel recall on G5 slides, FP G5 area on G5-free slides,
AUC, and sensitivity/specificity at Youden and at any-pixel (>0).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def annotated_pred_g5(r: dict) -> float:
    parts = [
        r.get("pred_g5_on_stroma"),
        r.get("pred_g5_on_normal"),
        r.get("pred_g5_on_g3"),
        r.get("pred_g5_on_g4"),
        r.get("pred_g5_on_g5"),
    ]
    if all(p not in (None, "") for p in parts):
        return sum(_f(p) for p in parts)
    return max(0.0, _f(r.get("pred_g5")) - _f(r.get("pred_g5_on_bg") or 0))


def scores(rows: list[dict]) -> tuple[list[int], list[float], list[dict]]:
    y, s, extra = [], [], []
    for r in rows:
        ann = _f(r.get("ann_px"))
        if not math.isfinite(ann) or ann <= 0:
            continue
        pred_ann = annotated_pred_g5(r)
        frac = pred_ann / ann
        label = int(_f(r.get("gt_has_g5") or r.get("gt_g5")) > 0)
        rec = pred_ann
        gt5 = _f(r.get("gt_g5") or 0)
        tp = _f(r.get("pred_g5_on_g5") or 0)
        y.append(label)
        s.append(frac)
        extra.append(
            {
                "slide_id": r["slide_id"],
                "gt_has_g5": label,
                "pred_g5_frac_ann": frac,
                "pred_g5_on_ann": pred_ann,
                "ann_px": ann,
                "gt_g5": gt5,
                "pixel_recall": (tp / gt5) if gt5 > 0 else float("nan"),
                "fp_g5_on_ann": pred_ann - tp,
            }
        )
    return y, s, extra


def auc_mann_whitney(y: list[int], s: list[float]) -> float | None:
    pos = [si for yi, si in zip(y, s) if yi == 1]
    neg = [si for yi, si in zip(y, s) if yi == 0]
    if not pos or not neg:
        return None
    better = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                better += 1
            elif p == n:
                ties += 1
    return (better + 0.5 * ties) / (len(pos) * len(neg))


def sens_spec(y: list[int], s: list[float], thr: float) -> tuple[float, float]:
    tp = fp = tn = fn = 0
    for yi, si in zip(y, s):
        pred = si > thr
        if yi and pred:
            tp += 1
        elif yi and not pred:
            fn += 1
        elif (not yi) and pred:
            fp += 1
        else:
            tn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def youden(y: list[int], s: list[float]) -> dict:
    cands = sorted(set(s))
    best = {"threshold": 0.0, "sens": float("nan"), "spec": float("nan"), "youden": -1.0}
    for thr in cands:
        # classify as G5-present if score > thr; also try >= via midpoints
        se, sp = sens_spec(y, s, thr)
        j = (se + sp - 1.0) if math.isfinite(se) and math.isfinite(sp) else -1.0
        if j > best["youden"]:
            best = {"threshold": thr, "sens": se, "spec": sp, "youden": j}
    return best


def summarize(name: str, rows: list[dict]) -> dict:
    y, s, extra = scores(rows)
    g5 = [e for e in extra if e["gt_has_g5"] == 1]
    nog5 = [e for e in extra if e["gt_has_g5"] == 0]
    recs = [e["pixel_recall"] for e in g5 if math.isfinite(e["pixel_recall"])]
    fps = [e["fp_g5_on_ann"] for e in nog5]
    fracs_pos = [e["pred_g5_frac_ann"] for e in g5]
    fracs_neg = [e["pred_g5_frac_ann"] for e in nog5]
    any_thr = youden(y, s)
    se0, sp0 = sens_spec(y, s, 0.0)
    return {
        "name": name,
        "n": len(extra),
        "n_g5": len(g5),
        "n_no_g5": len(nog5),
        "pixel_recall_on_g5_slides": {
            "n": len(recs),
            "mean": sum(recs) / len(recs) if recs else None,
            "values": recs,
        },
        "fp_g5_area_on_g5_free": {
            "n": len(fps),
            "mean": sum(fps) / len(fps) if fps else None,
            "total": sum(fps) if fps else 0.0,
        },
        "pred_g5_frac_ann": {
            "g5_slides_mean": sum(fracs_pos) / len(fracs_pos) if fracs_pos else None,
            "g5_free_mean": sum(fracs_neg) / len(fracs_neg) if fracs_neg else None,
        },
        "auc": auc_mann_whitney(y, s),
        "any_pixel_on_ann": {"threshold": 0.0, "sens": se0, "spec": sp0},
        "youden": any_thr,
        "slides": extra,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "G5 slide detector. Prefer --train-csvs + --test-csvs: lock Youden "
            "threshold on pooled Train, then report sens/spec on each Test CSV "
            "(Omar protocol). Plain positional csvs still print in-sample Youden "
            "for debugging only."
        )
    )
    p.add_argument("csvs", nargs="*", type=Path, help="Legacy: one report per CSV (in-sample Youden)")
    p.add_argument("--names", default="", help="Comma names aligned with csvs / test-csvs")
    p.add_argument(
        "--train-csvs",
        nargs="+",
        type=Path,
        default=None,
        help="Pool these for one Youden threshold (e.g. S1+S2+S3 Train)",
    )
    p.add_argument(
        "--test-csvs",
        nargs="+",
        type=Path,
        default=None,
        help="Apply the train-locked threshold once to each of these",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.train_csvs and args.test_csvs:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        if not names:
            names = [c.parent.name for c in args.test_csvs]
        if len(names) != len(args.test_csvs):
            raise SystemExit("--names must match --test-csvs length")
        train_rows: list[dict] = []
        for path in args.train_csvs:
            train_rows.extend(load_rows(path))
        y_tr, s_tr, _ = scores(train_rows)
        thr_info = youden(y_tr, s_tr)
        thr = float(thr_info["threshold"])
        se_tr, sp_tr = sens_spec(y_tr, s_tr, thr)
        payload = {
            "protocol": "Youden thr locked on pooled --train-csvs; applied to each --test-csvs",
            "train": {
                "n": len(y_tr),
                "n_g5": int(sum(y_tr)),
                "n_no_g5": int(len(y_tr) - sum(y_tr)),
                "auc": auc_mann_whitney(y_tr, s_tr),
                "youden_threshold": thr,
                "sens": se_tr,
                "spec": sp_tr,
                "sources": [str(p) for p in args.train_csvs],
            },
            "test": [],
        }
        print(
            f"TRAIN pool n={payload['train']['n']} G5={payload['train']['n_g5']} "
            f"AUC={payload['train']['auc']} LOCKED thr={thr:.6g} "
            f"sens={se_tr:.3f} spec={sp_tr:.3f}",
            flush=True,
        )
        for path, name in zip(args.test_csvs, names):
            rows = load_rows(path)
            y, s, extra = scores(rows)
            se, sp = sens_spec(y, s, thr)
            ins = youden(y, s)
            rec = summarize(name, rows)
            entry = {
                "name": name,
                "n": len(y),
                "n_g5": int(sum(y)),
                "n_no_g5": int(len(y) - sum(y)),
                "auc": auc_mann_whitney(y, s),
                "threshold_from_train": thr,
                "sens": se,
                "spec": sp,
                "in_sample_youden_DO_NOT_USE": ins,
                "pixel_recall_mean_on_g5_slides": rec["pixel_recall_on_g5_slides"]["mean"],
                "slides": extra,
            }
            payload["test"].append(entry)
            print(
                f"{name}: n={entry['n']} G5={entry['n_g5']} AUC={entry['auc']} "
                f"TrainThr sens={se:.3f} spec={sp:.3f} "
                f"(in-sample Youden {ins['sens']:.3f}/{ins['spec']:.3f} — do not use)",
                flush=True,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        flat = args.out.with_suffix(".slides.csv")
        with flat.open("w", newline="") as f:
            w = None
            for entry in payload["test"]:
                for sl in entry["slides"]:
                    row = {
                        "model_subset": entry["name"],
                        "thr_from_train": thr,
                        "pred_g5_slide": int(sl["pred_g5_frac_ann"] > thr),
                        **sl,
                    }
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        w.writeheader()
                    w.writerow(row)
        print(f"wrote {args.out} and {flat}", flush=True)
        return

    if not args.csvs:
        raise SystemExit("Provide positional csvs, or --train-csvs with --test-csvs")
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if not names:
        names = [c.parent.parent.name + "/" + c.parent.name for c in args.csvs]
    reports = []
    for path, name in zip(args.csvs, names):
        reports.append(summarize(name, load_rows(path)))
        print(
            f"{name}: n={reports[-1]['n']} G5={reports[-1]['n_g5']} "
            f"AUC={reports[-1]['auc']} Youden thr={reports[-1]['youden']['threshold']:.4g} "
            f"sens={reports[-1]['youden']['sens']} spec={reports[-1]['youden']['spec']} "
            f"(IN-SAMPLE — prefer --train-csvs/--test-csvs)",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2))
    # flat CSV
    flat = args.out.with_suffix(".slides.csv")
    with flat.open("w", newline="") as f:
        w = None
        for rep in reports:
            for sl in rep["slides"]:
                row = {"model_subset": rep["name"], **sl}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow(row)
    print(f"wrote {args.out} and {flat}", flush=True)


if __name__ == "__main__":
    main()
