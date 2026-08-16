#!/usr/bin/env python3
"""CPU tests for epoch-eval watcher + L_slide-vs-Dice classifier + scorecard merge."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from classify_lslide_vs_dice import classify_step, summarize  # noqa: E402
from summarize_epoch_eval import build_summary, upsert_scorecard  # noqa: E402
from watch_opt3_epoch_eval import detect_new, ensure_baselines, eval_status  # noqa: E402


def test_fighting_together_flat() -> None:
    prev = {"epoch": 11, "cancer_dice": 0.504, "L_slide": 1.540, "L_pixel": 0.363}
    fight = classify_step(prev, {"epoch": 12, "cancer_dice": 0.457, "L_slide": 1.449, "L_pixel": 0.343})
    assert fight["pattern"] == "fighting", fight

    both = classify_step(
        {"epoch": 14, "cancer_dice": 0.537, "L_slide": 1.588, "L_pixel": 0.352},
        {"epoch": 15, "cancer_dice": 0.579, "L_slide": 1.426, "L_pixel": 0.348},
    )
    assert both["pattern"] == "together", both

    flat = classify_step(
        {"epoch": 26, "cancer_dice": 0.542, "L_slide": 1.455, "L_pixel": 0.342},
        {"epoch": 27, "cancer_dice": 0.546, "L_slide": 1.475, "L_pixel": 0.340},
    )
    assert flat["pattern"] == "flat", flat

    summary = summarize([fight, both, flat])
    assert summary["counts"]["fighting"] == 1
    assert summary["counts"]["together"] == 1
    assert summary["counts"]["flat"] == 1
    print("classify_ok")


def test_watcher_no_backfill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt_dir = tmp_p / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch_027_cancer_0.5465.pth").write_bytes(b"x")
        cfg = {"eval_root": str(tmp_p / "eval")}
        target = {"tag": "live", "ckpt_dir": str(ckpt_dir), "train_log": ""}
        state: dict = {}
        ensure_baselines(state, [target])
        assert state["baseline_epochs"]["live"] == [27]
        found = detect_new(state, [target], cfg)
        assert found == []

        (ckpt_dir / "epoch_028_cancer_0.5500.pth").write_bytes(b"x")
        found = detect_new(state, [target], cfg)
        assert len(found) == 1 and found[0]["epoch"] == 28

        dest = Path(found[0]["eval_dir"])
        dest.mkdir(parents=True)
        (dest / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        assert eval_status(dest) == "complete"
        found2 = detect_new(state, [target], cfg)
        assert found2 == []
    print("watcher_ok")


def test_scorecard_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = tmp_p / "epoch_028_cancer_0.5500.pth"
        ckpt.write_bytes(b"x")
        log = tmp_p / "training_log.csv"
        log.write_text(
            "epoch,train_loss,val_loss,cancer_dice,mean_dice,L_pixel,L_slide,L_grade,lr\n"
            "28,1.0,0.3,0.5500,0.67,0.34,1.40,0.90,1e-4\n",
            encoding="utf-8",
        )
        labeled = tmp_p / "plus.csv"
        labeled.write_text(
            "class,dice,iou,precision,recall\n"
            "G5,0.5,0.3,0.51,0.6\n"
            "cancer_dice,0.56,,,\n",
            encoding="utf-8",
        )
        panda_isup = tmp_p / "panda_isup_summary.json"
        panda_isup.write_text(json.dumps({"match_rate": 0.61}), encoding="utf-8")
        plus_isup = tmp_p / "plus_isup_summary.json"
        plus_isup.write_text(json.dumps({"isup_match_rate": 0.54}), encoding="utf-8")
        row = build_summary(
            tag="live",
            ckpt=ckpt,
            train_log=log,
            panda_isup_summary=panda_isup,
            panda_plus_labeled=labeled,
            panda_plus_isup_summary=plus_isup,
            job_id="1",
        )
        assert row["status"] == "complete"
        assert abs(row["panda_val_cancer_dice"] - 0.55) < 1e-9
        assert abs(row["panda_plus_cancer_dice"] - 0.56) < 1e-9
        assert abs(row["panda_plus_g5_precision"] - 0.51) < 1e-9
        assert abs(row["panda_isup_match"] - 0.61) < 1e-9
        scorecard = tmp_p / "scorecard.csv"
        upsert_scorecard(scorecard, row)
        upsert_scorecard(scorecard, {**row, "panda_plus_cancer_dice": 0.57})
        with scorecard.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert abs(float(rows[0]["panda_plus_cancer_dice"]) - 0.57) < 1e-9
    print("scorecard_ok")


def main() -> None:
    test_fighting_together_flat()
    test_watcher_no_backfill()
    test_scorecard_merge()
    print("ALL_PASS")


if __name__ == "__main__":
    main()
