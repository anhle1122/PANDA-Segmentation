#!/usr/bin/env python3
"""Read-only teacher-epoch gate for Omar-6 Opt3 runs.

Qualify an epoch only if all of:
  1. PANDA+ cancer Dice >= 0.58 (external bar; not in-domain val)
  2. L_slide <= L_slide from 3 epochs prior (falling or flat)
  3. PANDA+ G5 precision within --g5-precision-tol of ep7 (0.569370)
  4. named epoch_XXX_*.pth still on disk

PANDA+ Dice / G5 come from labeled eval CSVs and/or
outputs/docs/opt3_this_run/epoch_external_scorecard.csv.
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
DEFAULT_EXTERNAL = PROJECT / "outputs" / "docs" / "opt3_this_run" / "epoch_external_scorecard.csv"
EVAL_DIR = PROJECT / "outputs" / "evaluation"
EP7_LABELED = (
    EVAL_DIR
    / "uni2_upernet_raw_panda_plus_uni2_upernet_raw_opt3_omar6_grouped_soft01_best_labeled.csv"
)
EP7_G5_PRECISION = 0.569370
PANDA_PLUS_CANCER_MIN = 0.58
L_SLIDE_LOOKBACK = 3


def _f(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_train_log(path: Path) -> list[dict]:
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
            "cancer_dice": _f(row, "cancer_dice"),  # in-domain val (info only)
            "L_slide": _f(row, "L_slide"),
            "L_pixel": _f(row, "L_pixel"),
            "L_grade": _f(row, "L_grade"),
            "mean_dice": _f(row, "mean_dice"),
        }
    return [by_ep[k] for k in sorted(by_ep)]


def load_external_scorecard(path: Path) -> dict[int, dict]:
    """epoch -> {panda_plus_cancer_dice, panda_plus_g5_precision, ...}"""
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ep = int(float(row["epoch"]))
            except (KeyError, ValueError, TypeError):
                continue
            out[ep] = {
                "panda_plus_cancer_dice": _f(row, "panda_plus_cancer_dice"),
                "panda_plus_g5_precision": _f(row, "panda_plus_g5_precision"),
                "panda_plus_isup_match": _f(row, "panda_plus_isup_match"),
                "src": "epoch_external_scorecard.csv",
            }
    return out


def metrics_from_labeled_csv(path: Path) -> tuple[float | None, float | None]:
    """Return (cancer_dice, g5_precision) from a PANDA+ labeled metrics CSV."""
    if not path.is_file():
        return None, None
    cancer = None
    g5 = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = str(row.get("class", "")).strip()
            if name == "cancer_dice":
                cancer = _f(row, "dice")
            elif name.upper() == "G5":
                g5 = _f(row, "precision")
    return cancer, g5


def named_ckpt(ckpt_dir: Path, epoch: int) -> Path | None:
    hits = sorted(ckpt_dir.glob(f"epoch_{epoch:03d}_cancer_*.pth"))
    return hits[-1] if hits else None


def find_panda_plus_metrics(
    epoch: int,
    *,
    val_cancer: float | None,
    external: dict[int, dict],
) -> tuple[float | None, float | None, str]:
    """Return (panda_plus_cancer_dice, g5_precision, source_note)."""
    if epoch == 7:
        cancer, g5 = metrics_from_labeled_csv(EP7_LABELED)
        if cancer is not None or g5 is not None:
            return cancer, g5 if g5 is not None else EP7_G5_PRECISION, EP7_LABELED.name
        return None, EP7_G5_PRECISION, "hardcoded_ep7_g5_only"

    ext = external.get(epoch)
    if ext and ext.get("panda_plus_cancer_dice") is not None:
        return (
            ext["panda_plus_cancer_dice"],
            ext.get("panda_plus_g5_precision"),
            str(ext.get("src", "external_scorecard")),
        )

    matches = sorted(
        EVAL_DIR.glob(f"uni2_upernet_raw_panda_plus_epoch_{epoch:03d}_cancer_*_labeled.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if val_cancer is not None:
        close = []
        for p in matches:
            try:
                stem_c = float(p.name.split("cancer_")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if abs(stem_c - val_cancer) <= 0.005:
                close.append(p)
        if close:
            matches = close
    if not matches:
        return None, None, "no PANDA+ labeled CSV / external scorecard row"
    cancer, g5 = metrics_from_labeled_csv(matches[0])
    return cancer, g5, matches[0].name


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def assess(
    rows: list[dict],
    *,
    plus_min: float,
    lookback: int,
    g5_ref: float,
    g5_tol: float,
    ckpt_dir: Path,
    external: dict[int, dict],
) -> list[dict]:
    by_ep = {int(r["epoch"]): r for r in rows}
    out = []
    for r in rows:
        ep = int(r["epoch"])
        val_cancer = r["cancer_dice"]
        l_slide = r["L_slide"]
        prior = by_ep.get(ep - lookback)
        prior_l = prior["L_slide"] if prior else None
        ckpt = named_ckpt(ckpt_dir, ep)
        plus_dice, g5, plus_src = find_panda_plus_metrics(
            ep, val_cancer=val_cancer, external=external
        )

        plus_ok = plus_dice is not None and plus_dice >= plus_min
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
            g5_note = plus_src if plus_dice is None else "G5 missing in labeled CSV"
        else:
            gap = abs(g5 - g5_ref)
            g5_ok = gap <= g5_tol
            g5_note = (
                f"{g5:.3f} ({'within' if g5_ok else 'outside'} {g5_tol:.3f} of ep7 {g5_ref:.3f}; {plus_src})"
            )

        has_ckpt = ckpt is not None
        qualifies = bool(plus_ok and l_ok is True and g5_ok and has_ckpt)
        out.append(
            {
                "epoch": ep,
                "val_cancer": val_cancer,
                "plus_dice": plus_dice,
                "plus_ok": plus_ok,
                "plus_src": plus_src,
                "l_slide": l_slide,
                "l_ok": l_ok,
                "l_note": l_note,
                "g5": g5,
                "g5_ok": g5_ok,
                "g5_note": g5_note,
                "has_ckpt": has_ckpt,
                "ckpt": str(ckpt) if ckpt else "",
                "qualifies": qualifies,
                "historical": bool(plus_ok and l_ok is True and g5_ok and not has_ckpt),
            }
        )
    return out


def describe(a: dict, *, plus_min: float) -> str:
    plus = a["plus_dice"]
    if plus is None:
        c_bit = f"PANDA+ cancer n/a ({a['plus_src']})"
    elif a["plus_ok"]:
        c_bit = f"PANDA+ cancer {plus:.3f} (>= {plus_min:.3f}, ok)"
    else:
        c_bit = f"PANDA+ cancer {plus:.3f} (need {plus_min:.3f}, gap {plus_min - plus:.3f})"

    val = a["val_cancer"]
    v_bit = f"in-domain val {fmt(val)}"

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
    return f"ep{a['epoch']}: {c_bit}; {v_bit}; {l_bit}; {g_bit}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only teacher-epoch selector (PANDA+ Dice gate)")
    ap.add_argument("--scorecard", type=Path, default=DEFAULT_LOG, help="training_log.csv")
    ap.add_argument("--external-scorecard", type=Path, default=DEFAULT_EXTERNAL)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument(
        "--panda-plus-cancer-min",
        type=float,
        default=PANDA_PLUS_CANCER_MIN,
        help="Minimum PANDA+ cancer Dice (default 0.58)",
    )
    ap.add_argument("--l-slide-lookback", type=int, default=L_SLIDE_LOOKBACK)
    ap.add_argument("--g5-ref", type=float, default=EP7_G5_PRECISION)
    ap.add_argument("--g5-precision-tol", type=float, default=0.03)
    args = ap.parse_args()

    rows = load_train_log(args.scorecard)
    if not rows:
        raise SystemExit(f"no epoch rows in {args.scorecard}")
    external = load_external_scorecard(args.external_scorecard)
    assessed = assess(
        rows,
        plus_min=args.panda_plus_cancer_min,
        lookback=args.l_slide_lookback,
        g5_ref=args.g5_ref,
        g5_tol=args.g5_precision_tol,
        ckpt_dir=args.ckpt_dir,
        external=external,
    )
    winners = [a for a in assessed if a["qualifies"]]
    historical = [a for a in assessed if a["historical"]]
    print(
        f"scorecard={args.scorecard} epochs={len(assessed)} "
        f"PANDA+_cancer>={args.panda_plus_cancer_min:.3f} "
        f"L_slide<=ep-3 G5 within {args.g5_precision_tol:.3f} of {args.g5_ref:.6f}"
    )
    latest = assessed[-1]
    print(f"LATEST {describe(latest, plus_min=args.panda_plus_cancer_min)}")
    if historical:
        h = historical[-1]
        print(
            f"HISTORICAL_ONLY ep{h['epoch']} (metrics pass, named ckpt gone) | "
            f"{describe(h, plus_min=args.panda_plus_cancer_min)}"
        )
    if winners:
        best = max(winners, key=lambda a: (a["plus_dice"] or -1.0, a["epoch"]))
        print(f"CANDIDATE ep{best['epoch']} | {describe(best, plus_min=args.panda_plus_cancer_min)}")
        print(f"CKPT {best['ckpt']}")
        for a in winners:
            if a is not best:
                print(f"ALSO ep{a['epoch']} | {describe(a, plus_min=args.panda_plus_cancer_min)}")
        return
    print("NO_CANDIDATE")
    near = [a for a in assessed if a["plus_ok"] and a["has_ckpt"]]
    if near and near[-1]["epoch"] != latest["epoch"]:
        print(
            f"closest keepable PANDA+ pass: "
            f"{describe(near[-1], plus_min=args.panda_plus_cancer_min)}"
        )
    elif latest["plus_dice"] is not None and not latest["plus_ok"]:
        print(
            f"latest PANDA+ gap={args.panda_plus_cancer_min - latest['plus_dice']:.3f} "
            f"(in-domain val={fmt(latest['val_cancer'])})"
        )


if __name__ == "__main__":
    main()
