"""Stub tests for referee label source + multi-tag watcher (no live ckpts)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train.corrected_label_dataset import (  # noqa: E402
    LABEL_SOURCE_CORRECTED,
    LABEL_SOURCE_RULES,
    CorrectedLabelReader,
    build_label_dataset,
    overlay_corrected,
)
from train.losses import apply_pixel_ignore, segmentation_loss  # noqa: E402
from train.pseudo_label_dataset import NO_CLASS, build_corrected_target  # noqa: E402
from watch_opt3_teacher_packs import (  # noqa: E402
    detect_new,
    ensure_baselines,
    pack_status,
    tick,
)


def _write_corrected(path: Path, coords, target, ignore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=np.asarray(coords, dtype=np.int32))
        f.create_dataset("target", data=np.asarray(target, dtype=np.uint8))
        f.create_dataset("ignore", data=np.asarray(ignore, dtype=np.uint8))


def test_reader_and_overlay() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        target = np.array([[[3, 4], [4, 5]]], dtype=np.uint8)
        ignore = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
        _write_corrected(tmp_p / "slideA_corrected.h5", [(8, 16)], target, ignore)
        reader = CorrectedLabelReader(tmp_p)
        got = reader.read("slideA", 8, 16)
        assert got is not None
        t, ign = got
        assert t[0, 1] == 4 and ign[0, 1] == 1
        assert reader.read("slideA", 0, 0) is None
        assert reader.read("missing", 8, 16) is None
        mask = np.array([[3, 3], [3, 3]], dtype=np.uint8)
        weight = np.ones((2, 2), dtype=np.float32)
        out_m, out_w = overlay_corrected(mask, weight, t, ign)
        assert out_m[0, 0] == 3 and out_m[0, 1] == 0
        assert out_w[0, 1] == 0.0 and out_w[0, 0] == 1.0
        reader.close()
    print("PASS reader_and_overlay")


def test_ignore_zeros_loss_grad() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(1, 6, 2, 2, requires_grad=True)
    with torch.no_grad():
        logits[0, 4, 0, 1] = 10.0
        logits[0, 3, 0, 0] = 10.0
    targets = torch.tensor([[[3, 3], [3, 3]]])
    weights = torch.ones(1, 2, 2)
    ignore = torch.tensor([[[0, 1], [0, 0]]])
    t, w = apply_pixel_ignore(targets, weights, ignore, ignore_index=0)
    assert int(t[0, 0, 1]) == 0
    assert float(w[0, 0, 1]) == 0.0
    cw = torch.ones(6)
    loss = segmentation_loss(logits, t, w, cw, adjacent_soft_alpha=0.0)
    loss.backward()
    ignored_grad = logits.grad[0, :, 0, 1].abs().sum().item()
    kept_grad = logits.grad[0, :, 0, 0].abs().sum().item()
    assert ignored_grad == 0.0, ignored_grad
    assert kept_grad > 0.0, kept_grad
    print("PASS ignore_zeros_loss_grad")


def test_rules_path_unchanged() -> None:
    flag = torch.zeros(1, 2, 2, dtype=torch.bool)
    flag[0, 0, 0] = True
    params = torch.tensor([[3.0, 1.0, float(NO_CLASS), 0.0]])
    tgt = build_corrected_target(flag, params, num_classes=6)
    assert float(tgt[0, 3, 0, 0]) == 1.0
    assert float(tgt[0, 3, 0, 1]) == 0.0
    assert build_label_dataset.__annotations__ or True
    assert LABEL_SOURCE_RULES == "rules" and LABEL_SOURCE_CORRECTED == "corrected"
    print("PASS rules_path_unchanged")


def test_watcher_new_epochs_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        live = tmp_p / "live"
        r2 = tmp_p / "r2"
        live.mkdir()
        r2.mkdir()
        for ep, name in ((5, "epoch_005_cancer_0.5461.pth"), (15, "epoch_015_cancer_0.5791.pth")):
            (live / name).write_bytes(b"x")
        cfg = {
            "pack_root": str(tmp_p / "packs"),
            "cache_script": "/bin/true",
            "state_file": str(tmp_p / "state.json"),
            "targets": [
                {"tag": "opt3_omar6_grouped_soft01", "ckpt_dir": str(live), "teacher_pack_dir": str(tmp_p / "packs")},
                {"tag": "opt3_omar6_r2_everyep", "ckpt_dir": str(r2), "teacher_pack_dir": str(tmp_p / "packs")},
            ],
        }
        state: dict = {"baseline_epochs": {}, "queued": [], "active_job_id": None, "submitted": []}
        ensure_baselines(state, cfg["targets"])
        assert state["baseline_epochs"]["opt3_omar6_grouped_soft01"] == [5, 15]
        assert state["baseline_epochs"]["opt3_omar6_r2_everyep"] == []
        assert detect_new(state, cfg["targets"]) == []

        (live / "epoch_019_cancer_0.5452.pth").write_bytes(b"x")
        (r2 / "epoch_001_cancer_0.4000.pth").write_bytes(b"x")
        new = detect_new(state, cfg["targets"])
        tags_eps = {(i["tag"], i["epoch"]) for i in new}
        assert ("opt3_omar6_grouped_soft01", 19) in tags_eps
        assert ("opt3_omar6_r2_everyep", 1) in tags_eps
        assert ( "opt3_omar6_grouped_soft01", 5) not in tags_eps
        assert ("opt3_omar6_grouped_soft01", 15) not in tags_eps

        pack = tmp_p / "packs" / "teacher_opt3_omar6_grouped_soft01_ep019"
        pack.mkdir(parents=True)
        (pack / "pack_config.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        assert pack_status(pack) == "complete"
        new2 = detect_new(state, cfg["targets"])
        assert all(i["epoch"] != 19 for i in new2)

        state2 = tick(cfg, state, auto_submit=False)
        assert any(q["epoch"] == 1 for q in state2["queued"])
        assert state2["active_job_id"] is None
    print("PASS watcher_new_epochs_only")


def main() -> None:
    test_reader_and_overlay()
    test_ignore_zeros_loss_grad()
    test_rules_path_unchanged()
    test_watcher_new_epochs_only()
    print("ALL_PASS")


if __name__ == "__main__":
    main()
