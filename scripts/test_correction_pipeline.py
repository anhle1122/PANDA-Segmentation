#!/usr/bin/env python3
"""CPU tests for parameterized correction identity, gates, and three-way referee."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from apply_isup_referee import apply_slide  # noqa: E402
from correction_pipeline import (  # noqa: E402
    FLAG_DICE_UP_ISUP_NOT,
    FLAG_MISSING_ISUP,
    STATUS_NOT_VALIDATED,
    STATUS_VALIDATED,
    assess_validation,
    comparison_rows,
    corrections_dir,
    g5_bias_from_report,
    identity_entry,
    model_id,
    pack_exists_for_tag_epoch,
    pack_tag,
    parse_epoch_from_ckpt,
    teacher_pack_dir,
    write_comparison,
)


def test_paths_include_recipe_and_stay_separate() -> None:
    a = teacher_pack_dir("opt3_omar6_grouped_soft01", "pre_lora_fix", 29)
    b = teacher_pack_dir("opt3_omar6_grouped_soft01", "post_lora_fix", 29)
    c = corrections_dir("opt3_omar6_lambda015", "post_lora_fix_lambda015", 5)
    assert "pre_lora_fix" in a.name
    assert "post_lora_fix" in b.name
    assert a != b
    assert a.name == "teacher_opt3_omar6_grouped_soft01_pre_lora_fix_ep029"
    assert c.name == "corrections_opt3_omar6_lambda015_post_lora_fix_lambda015_ep005"
    assert pack_tag("opt3_omar6_grouped_soft01", "pre_lora_fix") == "opt3_omar6_grouped_soft01_pre_lora_fix"
    assert model_id("opt3_omar6_grouped_soft01", "pre_lora_fix", 29).endswith("_ep029")
    ckpt = Path("epoch_029_cancer_0.5499.pth")
    assert parse_epoch_from_ckpt(ckpt) == 29
    print("PASS paths_include_recipe_and_stay_separate")


def test_pack_exists_skip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "teacher_x_pre_lora_fix_ep029"
        assert pack_exists_for_tag_epoch(d) is False
        d.mkdir()
        assert pack_exists_for_tag_epoch(d) is False
        (d / "slide_srcpred.h5").write_bytes(b"x")
        assert pack_exists_for_tag_epoch(d) is True
        e = Path(tmp) / "teacher_y"
        e.mkdir()
        (e / "pack_config.json").write_text("{}", encoding="utf-8")
        assert pack_exists_for_tag_epoch(e) is True
    print("PASS pack_exists_skip")


def test_identity_records_commit_and_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "epoch_029_cancer_0.5499.pth"
        ckpt.write_bytes(b"x")
        entry = identity_entry(
            checkpoint=ckpt,
            run_tag="opt3_omar6_grouped_soft01",
            recipe_version="pre_lora_fix",
            source_job_id="5443101",
            code_commit="488c5e7",
            code_commit_note="in-memory LoRA not in AdamW",
            epoch=29,
        )
        assert entry["checkpoint_filename"] == "epoch_029_cancer_0.5499.pth"
        assert entry["source_job_id"] == "5443101"
        assert entry["code_commit"] == "488c5e7"
        assert entry["recipe_version"] == "pre_lora_fix"
        assert entry["auto_train"] is False
        assert "pre_lora_fix" in entry["paths"]["teacher_pack"]
        assert "pre_lora_fix" in entry["paths"]["corrections"]
    print("PASS identity_records_commit_and_job")


def test_dice_up_isup_not_blocks() -> None:
    status, flags = assess_validation(
        {"panda_plus_cancer_dice": 0.609, "panda_plus_isup_match": 0.50},
        {"panda_plus_cancer_dice": 0.563, "panda_plus_isup_match": 0.58},
    )
    assert status == STATUS_NOT_VALIDATED
    assert FLAG_DICE_UP_ISUP_NOT in flags
    status2, flags2 = assess_validation(
        {"panda_plus_cancer_dice": 0.609, "panda_plus_isup_match": 0.604},
        {"panda_plus_cancer_dice": 0.563, "panda_plus_isup_match": None},
    )
    assert status2 == STATUS_VALIDATED
    assert FLAG_DICE_UP_ISUP_NOT not in flags2
    status3, flags3 = assess_validation({"panda_plus_cancer_dice": 0.609})
    assert status3 == STATUS_NOT_VALIDATED
    assert FLAG_MISSING_ISUP in flags3
    print("PASS dice_up_isup_not_blocks")


def test_three_way_referee() -> None:
    mask = np.array([[[3, 3, 3, 3]]], dtype=np.uint8)
    pred = np.array([[[3, 5, 5, 4]]], dtype=np.uint8)
    # agree, high-conf illegal G5, low-conf illegal G5, high-conf legal G4
    maxprob = np.array([[[0.9, 0.9, 0.2, 0.9]]], dtype=np.float32)
    allowed = {3, 4}
    target, ignore, counts = apply_slide(pred, maxprob, mask, allowed, tau=0.7)
    assert int(target[0, 0, 0]) == 3
    assert int(ignore[0, 0, 0]) == 0
    assert int(target[0, 0, 1]) == 4  # nearest allowed to G5
    assert int(ignore[0, 0, 1]) == 0
    assert int(target[0, 0, 2]) == 3  # low conf: label unchanged
    assert int(ignore[0, 0, 2]) == 1
    assert int(target[0, 0, 3]) == 3  # legal fight: keep mask
    assert int(ignore[0, 0, 3]) == 0
    assert counts["n_swap"] == 1
    assert counts["n_ignore"] == 1
    print("PASS three_way_referee")


def test_g5_bias_and_comparison_not_autolearn() -> None:
    report = {
        "g5_bias_flag": True,
        "g5_summary": {
            "pct_high_conf_swaps_to_g5": 40.0,
            "n_swap_to_g5": 40,
            "n_high_conf_swap": 100,
            "pct_original_mask_g5": 2.0,
            "n_original_mask_g5": 20,
            "n_original_mask_pixels": 1000,
        },
    }
    g5 = g5_bias_from_report(report)
    assert g5["g5_bias_flag"] is True
    assert abs(g5["ratio_swap_g5_vs_mask_g5"] - 20.0) < 1e-6
    payload = {
        "updated_at": "now",
        "models": {
            "opt3_omar6_grouped_soft01_pre_lora_fix_ep029": {
                "run_tag": "opt3_omar6_grouped_soft01",
                "recipe_version": "pre_lora_fix",
                "epoch": 29,
                "source_job_id": "5443101",
                "code_commit": "488c5e7",
                "metrics": {
                    "panda_plus_cancer_dice": 0.609,
                    "panda_plus_isup_match": 0.604,
                    "panda_plus_g5_precision": 0.559,
                    "panda_plus_g5_recall": 0.606,
                },
                "g5_bias": g5,
                "validation_status": "VALIDATED",
                "validation_flags": [],
                "paths": {"teacher_pack": "p", "corrections": "c"},
            }
        },
    }
    rows = comparison_rows(payload)
    assert rows[0]["auto_train"] is False
    assert rows[0]["recipe_version"] == "pre_lora_fix"
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "c.csv"
        md_path = Path(tmp) / "c.md"
        write_comparison(payload, csv_path=csv_path, md_path=md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "pre_lora_fix" in text
        assert "auto_train=false" in text
        assert "0.609" in text
    print("PASS g5_bias_and_comparison_not_autolearn")


def main() -> None:
    test_paths_include_recipe_and_stay_separate()
    test_pack_exists_skip()
    test_identity_records_commit_and_job()
    test_dice_up_isup_not_blocks()
    test_three_way_referee()
    test_g5_bias_and_comparison_not_autolearn()
    print("ALL_PASS")


if __name__ == "__main__":
    main()
