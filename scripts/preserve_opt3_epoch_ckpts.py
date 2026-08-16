#!/usr/bin/env python3
"""Copy each finished Opt3 epoch off latest.pth into an immutable epoch_*.pth.

Live trainer still overwrites latest.pth. At epoch end it writes training_log.csv
and latest.pth within seconds. Mid-epoch flushes update latest.pth only.
Never overwrite an existing epoch_*.pth. Do not touch the H200 train job.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

CKPT_DIR = Path(
    "/common/omarmlab/members/anh/panda_project/outputs/checkpoints/"
    "uni2_upernet_raw_opt3_omar6_grouped_soft01"
)
# Epoch-end writes log + latest together. Mid-epoch only touches latest.
END_WINDOW_SEC = 90.0


def _log_epochs(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.is_file():
        return []
    rows: list[tuple[int, float]] = []
    with log_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((int(float(row["epoch"])), float(row["cancer_dice"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _has_epoch_file(ckpt_dir: Path, epoch: int) -> bool:
    return any(ckpt_dir.glob(f"epoch_{epoch:03d}_*.pth"))


def snapshot_finished(ckpt_dir: Path) -> list[str]:
    done: list[str] = []
    latest = ckpt_dir / "latest.pth"
    log_path = ckpt_dir / "training_log.csv"
    log_rows = _log_epochs(log_path)
    if not log_rows or not latest.is_file() or latest.stat().st_size < 1_000_000:
        return done
    ep, cancer = log_rows[-1]
    if _has_epoch_file(ckpt_dir, ep):
        return done
    dt = abs(latest.stat().st_mtime - log_path.stat().st_mtime)
    if dt > END_WINDOW_SEC:
        return done
    dest = ckpt_dir / f"epoch_{ep:03d}_cancer_{cancer:.4f}.pth"
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copy2(latest, tmp)
    os.replace(tmp, dest)
    done.append(str(dest))
    print(
        f"preserved {dest.name} (log↔latest Δ={dt:.1f}s cancer={cancer:.4f})",
        flush=True,
    )
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--interval-sec", type=int, default=15)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    print(
        f"watching {args.ckpt_dir} every {args.interval_sec}s "
        f"(copy when log and latest.pth update together)",
        flush=True,
    )
    while True:
        try:
            snapshot_finished(args.ckpt_dir)
        except Exception as exc:
            print(f"warn: {exc}", flush=True)
        if args.once:
            break
        time.sleep(max(10, int(args.interval_sec)))


if __name__ == "__main__":
    main()
