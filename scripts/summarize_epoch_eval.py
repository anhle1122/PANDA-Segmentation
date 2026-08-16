#!/usr/bin/env python3
"""Merge train-log PANDA Dice with PANDA ISUP + PANDA+ Dice/ISUP into one row."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_([0-9]+(?:\.[0-9]+)?)")


def _f(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_ckpt_stem(ckpt: Path) -> tuple[int | None, float | None]:
    m = EPOCH_RE.search(ckpt.name)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def train_log_row(path: Path | None, epoch: int | None) -> dict:
    empty = {"cancer_dice": None, "L_slide": None, "L_pixel": None, "L_grade": None, "mean_dice": None}
    if path is None or epoch is None or not path.is_file():
        return empty
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                ep = int(float(raw["epoch"]))
            except (KeyError, ValueError, TypeError):
                continue
            if ep == epoch:
                return {
                    "cancer_dice": _f(raw, "cancer_dice"),
                    "L_slide": _f(raw, "L_slide"),
                    "L_pixel": _f(raw, "L_pixel"),
                    "L_grade": _f(raw, "L_grade"),
                    "mean_dice": _f(raw, "mean_dice"),
                }
    return empty


def labeled_metrics(path: Path | None) -> dict:
    out = {"panda_plus_cancer_dice": None, "panda_plus_g5_precision": None}
    if path is None or not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = str(raw.get("class", "")).strip()
            if name == "cancer_dice":
                out["panda_plus_cancer_dice"] = _f(raw, "dice")
            elif name.upper() == "G5":
                out["panda_plus_g5_precision"] = _f(raw, "precision")
    return out


def json_metric(path: Path | None, *keys: str) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return None


def build_summary(
    *,
    tag: str,
    ckpt: Path,
    train_log: Path | None,
    panda_isup_summary: Path | None,
    panda_plus_labeled: Path | None,
    panda_plus_isup_summary: Path | None,
    job_id: str | None = None,
) -> dict:
    epoch, ckpt_dice = parse_ckpt_stem(ckpt)
    train = train_log_row(train_log, epoch)
    plus = labeled_metrics(panda_plus_labeled)
    panda_isup = json_metric(panda_isup_summary, "match_rate")
    plus_isup = json_metric(panda_plus_isup_summary, "isup_match_rate", "match_rate")
    row = {
        "tag": tag,
        "epoch": epoch,
        "ckpt": str(ckpt),
        "panda_val_cancer_dice": train["cancer_dice"] if train["cancer_dice"] is not None else ckpt_dice,
        "mean_dice": train["mean_dice"],
        "L_slide": train["L_slide"],
        "L_pixel": train["L_pixel"],
        "L_grade": train["L_grade"],
        "panda_isup_match": panda_isup,
        "panda_plus_cancer_dice": plus["panda_plus_cancer_dice"],
        "panda_plus_g5_precision": plus["panda_plus_g5_precision"],
        "panda_plus_isup_match": plus_isup,
        "job_id": job_id or "",
        "status": "complete"
        if panda_isup is not None and plus["panda_plus_cancer_dice"] is not None and plus_isup is not None
        else "partial",
    }
    return row


SCORECARD_FIELDS = [
    "tag",
    "epoch",
    "panda_val_cancer_dice",
    "panda_isup_match",
    "panda_plus_cancer_dice",
    "panda_plus_isup_match",
    "panda_plus_g5_precision",
    "L_slide",
    "L_pixel",
    "L_grade",
    "mean_dice",
    "status",
    "job_id",
    "ckpt",
]


def upsert_scorecard(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    key = (str(row.get("tag", "")), str(row.get("epoch", "")))
    kept = [r for r in existing if (r.get("tag", ""), r.get("epoch", "")) != key]
    kept.append({k: "" if row.get(k) is None else row.get(k) for k in SCORECARD_FIELDS})
    kept.sort(key=lambda r: (r.get("tag", ""), int(float(r.get("epoch") or 0))))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS)
        w.writeheader()
        w.writerows(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write per-epoch PANDA / PANDA+ eval summary")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--train-log", type=Path, default=None)
    ap.add_argument("--panda-isup-summary", type=Path, default=None)
    ap.add_argument("--panda-plus-labeled", type=Path, default=None)
    ap.add_argument("--panda-plus-isup-summary", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--scorecard", type=Path, required=True)
    ap.add_argument("--job-id", default="")
    args = ap.parse_args()
    row = build_summary(
        tag=args.tag,
        ckpt=args.ckpt,
        train_log=args.train_log,
        panda_isup_summary=args.panda_isup_summary,
        panda_plus_labeled=args.panda_plus_labeled,
        panda_plus_isup_summary=args.panda_plus_isup_summary,
        job_id=args.job_id or None,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    upsert_scorecard(args.scorecard, row)
    print(
        f"CHECK tag={row['tag']} ep={row['epoch']} "
        f"PANDA_dice={row['panda_val_cancer_dice']} PANDA_isup={row['panda_isup_match']} "
        f"PLUS_dice={row['panda_plus_cancer_dice']} PLUS_isup={row['panda_plus_isup_match']} "
        f"G5_prec={row['panda_plus_g5_precision']} status={row['status']}"
    )


if __name__ == "__main__":
    main()
