#!/usr/bin/env python3
"""Keep src/ + scripts/ alive across NFS wipes.

/common is often mounted RO; outputs/ is RW. This watcher:
1. Copies tracked src/scripts into outputs/_code_mirror when live is healthy.
2. If live canaries vanish, refreshes the mirror from git HEAD.
3. Tries to restore live files; if the mount is RO, the mirror stays the
   source of truth for PYTHONPATH / sbatch.
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
HOME_BAK = Path.home() / "panda_code_backup"
TREES = ("src", "scripts", ".cursor/rules")
CANARIES = (
    "src/evaluate.py",
    "src/train/uni2_upernet.py",
    "src/train_uni2_opt3_slidebag.py",
    "src/train/corrected_label_dataset.py",
    "scripts/slurm_train_opt3_round4_corrected.sh",
    "scripts/hpc_use_code.sh",
)
LABEL_SOURCE_MARKERS = ("--label-source", "--corrected-dir")
SKIP_NAMES = {"__pycache__", ".pytest_cache"}


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{ts} | {msg}", flush=True)


def git_tracked() -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(PROJECT), "ls-files", *TREES],
        text=True,
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def trainer_has_corrected_flags(root: Path) -> bool:
    trainer = root / "src" / "train_uni2_opt3_slidebag.py"
    if not trainer.is_file():
        return False
    text = trainer.read_text(encoding="utf-8", errors="replace")
    return all(mark in text for mark in LABEL_SOURCE_MARKERS)


def is_healthy(root: Path) -> bool:
    if not all((root / rel).is_file() and (root / rel).stat().st_size > 0 for rel in CANARIES):
        return False
    return trainer_has_corrected_flags(root)


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)
    try:
        shutil.copymode(src, dest)
    except OSError:
        pass


def same_file(a: Path, b: Path) -> bool:
    if not a.is_file() or not b.is_file():
        return False
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def sync_tree(src_root: Path, dest_root: Path, rels: list[str]) -> int:
    n = 0
    for rel in rels:
        src = src_root / rel
        dest = dest_root / rel
        if not src.is_file():
            continue
        if any(part in SKIP_NAMES for part in Path(rel).parts):
            continue
        if same_file(src, dest):
            continue
        copy_file(src, dest)
        n += 1
    return n


def git_blob(rel: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT), "show", f"HEAD:{rel}"])
    except subprocess.CalledProcessError:
        return None


def restore_from_git(dest_root: Path, rels: list[str]) -> int:
    n = 0
    for rel in rels:
        dest = dest_root / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        blob = git_blob(rel)
        if blob is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
        tmp.write_bytes(blob)
        os.replace(tmp, dest)
        n += 1
    return n


def try_restore_live(rels: list[str]) -> tuple[int, str]:
    ok = 0
    last_err = "none"
    for rel in rels:
        dest = PROJECT / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        src = MIRROR / rel
        try:
            if src.is_file() and src.stat().st_size > 0:
                copy_file(src, dest)
                ok += 1
                continue
            blob = git_blob(rel)
            if blob is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
            tmp.write_bytes(blob)
            os.replace(tmp, dest)
            ok += 1
        except OSError as exc:
            last_err = str(exc)
            return ok, last_err
    return ok, last_err


def write_heartbeat() -> None:
    path = PROJECT / "outputs" / "_code_mirror" / "HEARTBEAT"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{datetime.now().astimezone().isoformat()}\n"
        f"live_healthy={is_healthy(PROJECT)}\n"
        f"mirror_healthy={is_healthy(MIRROR)}\n",
        encoding="utf-8",
    )


def tick() -> None:
    try:
        rels = git_tracked()
    except subprocess.CalledProcessError as e:
        log(f"git ls-files failed: {e}")
        return
    live_ok = is_healthy(PROJECT)
    mirror_ok = is_healthy(MIRROR)
    if live_ok:
        n = sync_tree(PROJECT, MIRROR, rels)
        n2 = 0
        try:
            n2 = sync_tree(PROJECT, HOME_BAK, rels)
        except OSError as exc:
            log(f"home backup skip: {exc}")
        if n or n2:
            log(f"synced live -> mirror={n} home={n2}")
    elif not mirror_ok:
        n = restore_from_git(MIRROR, rels)
        log(f"WIPE live missing; restored {n} files from git -> {MIRROR}")
    else:
        log("WIPE live missing; mirror still healthy (jobs should use PANDA_CODE_SRC)")
        n = restore_from_git(MIRROR, rels)
        if n:
            log(f"filled {n} extra mirror files from git")
    if not live_ok:
        ok, err = try_restore_live(rels)
        log(f"live restore wrote={ok} err={err} live_healthy={is_healthy(PROJECT)}")
    write_heartbeat()


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
