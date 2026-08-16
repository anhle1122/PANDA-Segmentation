#!/usr/bin/env python3
"""Read-only teacher-epoch gate for the live Omar-6 run.

Qualify an epoch only if all of:
  1. val cancer Dice >= 0.579 (ep15 remaining-best bar)
  2. L_slide <= L_slide from 3 epochs prior (falling or flat)
  3. PANDA+ G5 precision within --g5-precision-tol of ep7 (0.569370)

Prints the most recent epoch's gaps when nothing qualifies.
Safe to re-run: stdout only, no writes, no Slurm submits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
DEFAULT_CKPT_DIR = (
    PROJECT
    / "outputs"
    / "checkpoints"
    / "uni2_upernet_raw_opt3_omar6_grouped_soft01"
)
DEFAULT_LOG = DEFAULT_CKPT_DIR / "training_log.csv"
EVAL_DIR = PROJECT / "outputs" / "evaluation"
EP7_LABELED = (
    EVAL_DIR
    / "uni2_upernet_raw_panda_plus_uni2_upernet_raw_opt3_omar6_grouped_soft01_best_labeled.csv"
)
EP7_G5_PRECISION = 0.569370
VAL_CANCER_MIN = 0.579
L_SLIDE_LOOKBACK = 3


def _f(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_scorecard(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"scorecard missing: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_ep: dict[int, dict] = {}
    for row in rows:
        try:
            ep = int(float(row["epoch"]))
        except (KeyError, ValueError, TypeError):
            continue
        by_ep[ep] = {
            "epoch": ep,
            "cancer_dice": _f(row, "cancer_dice"),
            "L_slide": _f(row, "L_slide"),
            "L_pixel": _f(row, "L_pixel"),
            "L_grade": _f(row, "L_grade"),
            "mean_dice": _f(row, "mean_dice"),
        }
    return [by_ep[k] for k in sorted(by_ep)]


def g5_precision_from_csv(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("class", "")).strip().upper() == "G5":
                try:
                    return float(row["precision"])
                except (KeyError, ValueError, TypeError):
                    return None
    return None


def named_ckpt(ckpt_dir: Path, epoch: int) -> Path | None:
    hits = sorted(ckpt_dir.glob(f"epoch_{epoch:03d}_cancer_*.pth"))
    return hits[-1] if hits else None


def find_g5_precision(epoch: int, cancer: float | None) -> tuple[float | None, str]:
    if epoch == 7:
        val = g5_precision_from_csv(EP7_LABELED)
        if val is not None:
            return val, str(EP7_LABELED.name)
        return EP7_G5_PRECISION, "hardcoded_ep7_ref"
    matches = sorted(
        EVAL_DIR.glob(f"uni2_upernet_raw_panda_plus_epoch_{epoch:03d}_cancer_*_labeled.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cancer is not None:
        close = []
        for p in matches:
            try:
                stem_c = float(p.name.split("cancer_")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if abs(stem_c - cancer) <= 0.005:
                close.append(p)
        if close:
            matches = close
    if not matches:
        return None, "not in scorecard (no PANDA+ labeled CSV)"
    val = g5_precision_from_csv(matches[0])
    return val, matches[0].name


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def assess(
    rows: list[dict],
    *,
    val_min: float,
    lookback: int,
    g5_ref: float,
    g5_tol: float,
    ckpt_dir: Path,
) -> list[dict]:
    by_ep = {int(r["epoch"]): r for r in rows}
    out = []
    for r in rows:
        ep = int(r["epoch"])
        cancer = r["cancer_dice"]
        l_slide = r["L_slide"]
        prior = by_ep.get(ep - lookback)
        prior_l = prior["L_slide"] if prior else None
        ckpt = named_ckpt(ckpt_dir, ep)
        g5, g5_src = find_g5_precision(ep, cancer)

        cancer_ok = cancer is not None and cancer >= val_min
        if prior_l is None:
            l_ok = None
            l_note = f"no ep{ep - lookback} yet"
        elif l_slide is None:
            l_ok = False
            l_note = "missing L_slide"
        else:
            l_ok = l_slide <= prior_l + 1e-12
            trend = "falling" if l_slide < prior_l else ("flat" if abs(l_slide - prior_l) < 1e-9 else "rising")
            l_note = f"{trend}, vs ep{ep - lookback} {prior_l:.3f}"

        if g5 is None:
            g5_ok = False
            g5_note = g5_src
        else:
            gap = abs(g5 - g5_ref)
            g5_ok = gap <= g5_tol
            g5_note = (
                f"{g5:.3f} ({'within' if g5_ok else 'outside'} {g5_tol:.3f} of ep7 {g5_ref:.3f}; {g5_src})"
            )

        has_ckpt = ckpt is not None
        qualifies = bool(cancer_ok and l_ok is True and g5_ok and has_ckpt)
        out.append(
            {
                "epoch": ep,
                "cancer": cancer,
                "cancer_ok": cancer_ok,
                "l_slide": l_slide,
                "l_ok": l_ok,
                "l_note": l_note,
                "g5": g5,
                "g5_ok": g5_ok,
                "g5_note": g5_note,
                "has_ckpt": has_ckpt,
                "ckpt": str(ckpt) if ckpt else "",
                "qualifies": qualifies,
                "historical": bool(cancer_ok and l_ok is True and g5_ok and not has_ckpt),
            }
        )
    return out


def describe(a: dict, *, val_min: float) -> str:
    cancer = a["cancer"]
    if cancer is None:
        c_bit = "val cancer n/a (need scorecard)"
    elif a["cancer_ok"]:
        c_bit = f"val cancer {cancer:.3f} (>= {val_min:.3f}, ok)"
    else:
        c_bit = f"val cancer {cancer:.3f} (need {val_min:.3f}, gap {val_min - cancer:.3f})"

    l_slide = a["l_slide"]
    if a["l_ok"] is True:
        l_bit = f"L_slide {fmt(l_slide)} ({a['l_note']}, ok)"
    elif a["l_ok"] is None:
        l_bit = f"L_slide {fmt(l_slide)} ({a['l_note']})"
    else:
        l_bit = f"L_slide {fmt(l_slide)} ({a['l_note']}, fail)"

    if a["g5"] is None:
        g_bit = f"G5 precision: missing ({a['g5_note']})"
    elif a["g5_ok"]:
        g_bit = f"G5 precision: {a['g5']:.3f} (within threshold)"
    else:
        g_bit = f"G5 precision: {a['g5_note']}"
    return f"ep{a['epoch']}: {c_bit}; {l_bit}; {g_bit}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only teacher-epoch selector")
    ap.add_argument("--scorecard", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--val-cancer-min", type=float, default=VAL_CANCER_MIN)
    ap.add_argument("--l-slide-lookback", type=int, default=L_SLIDE_LOOKBACK)
    ap.add_argument("--g5-ref", type=float, default=EP7_G5_PRECISION)
    ap.add_argument("--g5-precision-tol", type=float, default=0.03)
    args = ap.parse_args()

    rows = load_scorecard(args.scorecard)
    if not rows:
        raise SystemExit(f"no epoch rows in {args.scorecard}")
    assessed = assess(
        rows,
        val_min=args.val_cancer_min,
        lookback=args.l_slide_lookback,
        g5_ref=args.g5_ref,
        g5_tol=args.g5_precision_tol,
        ckpt_dir=args.ckpt_dir,
    )
    winners = [a for a in assessed if a["qualifies"]]
    historical = [a for a in assessed if a["historical"]]
    print(
        f"scorecard={args.scorecard} epochs={len(assessed)} "
        f"val_cancer>={args.val_cancer_min:.3f} "
        f"L_slide<=ep-3 G5 within {args.g5_precision_tol:.3f} of {args.g5_ref:.6f}"
    )
    latest = assessed[-1]
    print(f"LATEST {describe(latest, val_min=args.val_cancer_min)}")
    if historical:
        h = historical[-1]
        print(f"HISTORICAL_ONLY ep{h['epoch']} (metrics pass, named ckpt gone) | {describe(h, val_min=args.val_cancer_min)}")
    if winners:
        best = max(winners, key=lambda a: (a["cancer"] or -1.0, a["epoch"]))
        print(f"CANDIDATE ep{best['epoch']} | {describe(best, val_min=args.val_cancer_min)}")
        print(f"CKPT {best['ckpt']}")
        for a in winners:
            if a is not best:
                print(f"ALSO ep{a['epoch']} | {describe(a, val_min=args.val_cancer_min)}")
        return
    print("NO_CANDIDATE")
    near = [a for a in assessed if a["cancer_ok"] and a["has_ckpt"]]
    if near and near[-1]["epoch"] != latest["epoch"]:
        print(f"closest keepable val-cancer pass: {describe(near[-1], val_min=args.val_cancer_min)}")


if __name__ == "__main__":
    main()
