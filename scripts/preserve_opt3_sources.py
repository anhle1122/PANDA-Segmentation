#!/usr/bin/env python3
"""Keep src/ + scripts/ alive across NFS wipes.

/common is often mounted RO; outputs/ is RW. This watcher:
1. Copies tracked src/scripts into outputs/_code_mirror when live is healthy.
2. If canary files vanish, restores them from git HEAD or the mirror.
Does not git-commit. Does not touch training jobs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
MIRROR = PROJECT / "outputs" / "_code_mirror"
TREES = ("src", "scripts", ".cursor/rules")
CANARIES = (
    PROJECT / "src" / "evaluate.py",
    PROJECT / "src" / "train" / "uni2_upernet.py",
    PROJECT / "src" / "train_uni2_opt3_slidebag.py",
    PROJECT / "scripts" / "slurm_train_opt3_slidebag.sh",
    PROJECT / "scripts" / "slurm_eval_opt3_epoch.sh",
    PROJECT / "scripts" / "watch_opt3_epoch_eval.py",
)


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{ts} | {msg}", flush=True)


def git_tracked() -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(PROJECT), "ls-files", *TREES],
        text=True,
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def live_healthy() -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in CANARIES)


def copy_file(src: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)
    shutil.copymode(src, dest)
    return True


def sync_live_to_mirror(rels: list[str]) -> int:
    n = 0
    for rel in rels:
        src = PROJECT / rel
        dest = MIRROR / rel
        if not src.is_file():
            continue
        if dest.is_file() and dest.stat().st_size == src.stat().st_size:
            if int(dest.stat().st_mtime) == int(src.stat().st_mtime):
                continue
        copy_file(src, dest)
        n += 1
    return n


def restore_one(rel: str) -> str | None:
    dest = PROJECT / rel
    mirror = MIRROR / rel
    if dest.is_file() and dest.stat().st_size > 0:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mirror.is_file() and mirror.stat().st_size > 0:
        try:
            copy_file(mirror, dest)
            return "mirror"
        except OSError:
            pass
    try:
        data = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"HEAD:{rel}"])
        tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return "git"
    except (OSError, subprocess.CalledProcessError):
        return None


def restore_live(rels: list[str]) -> tuple[int, int]:
    ok = fail = 0
    for rel in rels:
        dest = PROJECT / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        src = restore_one(rel)
        if src:
            ok += 1
        else:
            fail += 1
    return ok, fail


def tick() -> None:
    try:
        rels = git_tracked()
    except subprocess.CalledProcessError as e:
        log(f"git ls-files failed: {e}")
        return
    if live_healthy():
        n = sync_live_to_mirror(rels)
        if n:
            log(f"mirror updated {n} files -> {MIRROR}")
        return
    missing = [str(p.relative_to(PROJECT)) for p in CANARIES if not p.is_file()]
    log(f"WIPE detected missing={missing}")
    ok, fail = restore_live(rels)
    log(f"restore wrote={ok} still_missing={fail} healthy={live_healthy()}")
    if live_healthy():
        n = sync_live_to_mirror(rels)
        log(f"mirror refreshed {n} files after restore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-sec", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    MIRROR.mkdir(parents=True, exist_ok=True)
    log(f"watching src/scripts every {args.interval_sec}s mirror={MIRROR}")
    tick()
    if args.once:
        return
    while True:
        time.sleep(max(5, args.interval_sec))
        tick()


if __name__ == "__main__":
    main()
