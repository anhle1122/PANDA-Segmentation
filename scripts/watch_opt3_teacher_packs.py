#!/usr/bin/env python3
"""When a new Omar-6 epoch_*.pth appears, optionally submit a teacher-pack cache.

Default: print only. Set AUTO_SUBMIT=1 to sbatch slurm_cache_teacher_pack.sh
on L40S. Never overwrites an existing pack. Does not touch the H200 train.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

CKPT_DIR = Path(
    "/common/omarmlab/members/anh/panda_project/outputs/checkpoints/"
    "uni2_upernet_raw_opt3_omar6_grouped_soft01"
)
PACK_ROOT = Path("/common/omarmlab/members/anh/panda_project/outputs/pseudo_label")
TAG = "opt3_omar6_grouped_soft01"
SCRIPT = Path("/common/omarmlab/members/anh/panda_project/scripts/slurm_cache_teacher_pack.sh")
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")


def pack_dir(epoch: int) -> Path:
    return PACK_ROOT / f"teacher_{TAG}_ep{epoch:03d}"


def existing_epochs() -> list[tuple[int, Path]]:
    found = []
    for p in sorted(CKPT_DIR.glob("epoch_*_cancer_*.pth")):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def main() -> None:
    interval = int(os.environ.get("INTERVAL_SEC", "60"))
    auto = os.environ.get("AUTO_SUBMIT", "0").strip() in {"1", "true", "yes"}
    print(
        f"watching {CKPT_DIR} every {interval}s auto_submit={auto}",
        flush=True,
    )
    submitted: set[int] = set()
    while True:
        for ep, ckpt in existing_epochs():
            dest = pack_dir(ep)
            cfg = dest / "pack_config.json"
            n_h5 = len(list(dest.glob("*_srcpred.h5"))) if dest.is_dir() else 0
            if cfg.is_file() or n_h5 > 0:
                continue
            if ep in submitted:
                continue
            print(f"NEW epoch file {ckpt.name} -> pack {dest}", flush=True)
            if auto:
                cmd = ["sbatch", "--gres=gpu:l40s:1", str(SCRIPT), str(ckpt)]
                print("SUBMIT", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=False)
                submitted.add(ep)
            else:
                print(f"  dry-run: sbatch --gres=gpu:l40s:1 {SCRIPT} {ckpt}", flush=True)
                submitted.add(ep)
        time.sleep(interval)


if __name__ == "__main__":
    main()
