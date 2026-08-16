#!/usr/bin/env python3
"""Classify epoch-to-epoch L_slide vs Dice / L_seg (L_pixel) co-movement.

Three Omar patterns:
  together  — both improve, or both worsen (bounce / LR, not fighting)
  fighting  — L_slide improves while Dice or L_seg degrades
  flat      — neither moved enough to diagnose competition

A fourth label `mixed` is used when only one side moved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
DEFAULT_LOG = (
    PROJECT
    / "outputs"
    / "checkpoints"
    / "uni2_upernet_raw_opt3_omar6_grouped_soft01"
    / "training_log.csv"
)

# Moves smaller than this are treated as noise (val Dice 20k-patch subset).
DICE_EPS = 0.015
LSLIDE_EPS = 0.04
LSEG_EPS = 0.008


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for raw in csv.DictReader(f):
            try:
                ep = int(float(raw["epoch"]))
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(
                {
                    "epoch": ep,
                    "cancer_dice": float(raw["cancer_dice"]),
                    "L_slide": float(raw["L_slide"]),
                    "L_pixel": float(raw["L_pixel"]),
                    "L_grade": float(raw.get("L_grade") or 0.0),
                    "mean_dice": float(raw.get("mean_dice") or 0.0),
                }
            )
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _moved(delta: float, eps: float) -> str:
    if delta <= -eps:
        return "down"
    if delta >= eps:
        return "up"
    return "flat"


def classify_step(
    prev: dict,
    cur: dict,
    *,
    dice_eps: float = DICE_EPS,
    lslide_eps: float = LSLIDE_EPS,
    lseg_eps: float = LSEG_EPS,
) -> dict:
    d_dice = cur["cancer_dice"] - prev["cancer_dice"]
    d_slide = cur["L_slide"] - prev["L_slide"]
    d_seg = cur["L_pixel"] - prev["L_pixel"]
    dice_dir = _moved(d_dice, dice_eps)
    slide_dir = _moved(d_slide, lslide_eps)
    seg_dir = _moved(d_seg, lseg_eps)

    slide_better = slide_dir == "down"
    slide_worse = slide_dir == "up"
    dice_better = dice_dir == "up"
    dice_worse = dice_dir == "down"
    seg_better = seg_dir == "down"
    seg_worse = seg_dir == "up"
    seg_or_dice_worse = dice_worse or seg_worse
    seg_or_dice_better = dice_better or seg_better

    if slide_dir == "flat" and dice_dir == "flat" and seg_dir == "flat":
        pattern = "flat"
    elif slide_better and seg_or_dice_worse:
        pattern = "fighting"
    elif (slide_better and seg_or_dice_better and not seg_or_dice_worse) or (
        slide_worse and (dice_worse or seg_worse) and not (dice_better or seg_better)
    ):
        pattern = "together"
    elif slide_dir == "flat" or (dice_dir == "flat" and seg_dir == "flat"):
        pattern = "mixed"
    else:
        # opposite-quality move that is not the classic fight (L_slide worse, Dice better)
        pattern = "together"

    return {
        "epoch": cur["epoch"],
        "from_epoch": prev["epoch"],
        "cancer_dice": cur["cancer_dice"],
        "L_slide": cur["L_slide"],
        "L_pixel": cur["L_pixel"],
        "d_dice": d_dice,
        "d_L_slide": d_slide,
        "d_L_pixel": d_seg,
        "dice_dir": dice_dir,
        "slide_dir": slide_dir,
        "seg_dir": seg_dir,
        "pattern": pattern,
        "note": _note(pattern, slide_better, dice_worse, seg_worse, dice_better, slide_worse),
    }


def _note(
    pattern: str,
    slide_better: bool,
    dice_worse: bool,
    seg_worse: bool,
    dice_better: bool,
    slide_worse: bool,
) -> str:
    if pattern == "fighting":
        bits = []
        if dice_worse:
            bits.append("Dice down")
        if seg_worse:
            bits.append("L_seg up")
        return "L_slide down + " + " / ".join(bits)
    if pattern == "together" and slide_better and dice_better:
        return "both improved"
    if pattern == "together" and slide_worse and dice_worse:
        return "both worsened (bounce)"
    if pattern == "flat":
        return "neither moved"
    if pattern == "mixed":
        return "only one side moved"
    return "co-moved (not the fight signature)"


def classify_run(rows: list[dict], *, from_epoch: int = 2) -> list[dict]:
    out = []
    by_ep = {int(r["epoch"]): r for r in rows}
    for r in rows:
        ep = int(r["epoch"])
        if ep < from_epoch:
            continue
        prev = by_ep.get(ep - 1)
        if prev is None:
            continue
        out.append(classify_step(prev, r))
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def summarize(steps: list[dict]) -> dict:
    counts = {"fighting": 0, "together": 0, "flat": 0, "mixed": 0}
    for s in steps:
        counts[s["pattern"]] = counts.get(s["pattern"], 0) + 1
    r = pearson([s["d_L_slide"] for s in steps], [s["d_dice"] for s in steps])
    # Positive corr(ΔL_slide, ΔDice): both numeric-down together = fight-ish
    # Negative corr: L_slide down while Dice up = healthy
    if not steps:
        verdict = "no steps"
    elif counts["flat"] >= max(counts.values()) and counts["flat"] >= len(steps) / 2:
        verdict = "flat"
    elif counts["fighting"] >= max(1, len(steps) // 3) and counts["fighting"] >= counts["together"]:
        verdict = "fighting"
    else:
        verdict = "together"
    return {
        "n_steps": len(steps),
        "counts": counts,
        "corr_dLslide_dDice": r,
        "verdict": verdict,
        "fighting_epochs": [s["epoch"] for s in steps if s["pattern"] == "fighting"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify L_slide vs Dice/L_seg co-movement")
    ap.add_argument("--scorecard", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--from-epoch", type=int, default=10, help="First epoch to score (default: λ_slide on)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = load_rows(args.scorecard)
    steps = classify_run(rows, from_epoch=args.from_epoch)
    summary = summarize(steps)
    print(
        f"from_epoch>={args.from_epoch} n={summary['n_steps']} "
        f"together={summary['counts']['together']} fighting={summary['counts']['fighting']} "
        f"flat={summary['counts']['flat']} mixed={summary['counts']['mixed']} "
        f"corr(ΔL_slide,ΔDice)={summary['corr_dLslide_dDice']!s} verdict={summary['verdict']}"
    )
    if summary["fighting_epochs"]:
        print("fighting_epochs", summary["fighting_epochs"])
    for s in steps:
        print(
            f"ep{s['from_epoch']:02d}->{s['epoch']:02d} {s['pattern']:9s} "
            f"dDice={s['d_dice']:+.3f} dLslide={s['d_L_slide']:+.3f} dLseg={s['d_L_pixel']:+.3f} "
            f"{s['note']}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "from_epoch",
            "epoch",
            "pattern",
            "cancer_dice",
            "L_slide",
            "L_pixel",
            "d_dice",
            "d_L_slide",
            "d_L_pixel",
            "note",
        ]
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(steps)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
