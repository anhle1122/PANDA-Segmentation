#!/usr/bin/env python3
"""Watch Round-2 redo ckpts and sbatch leak-split PANDA+ (all/clean-30/leaked) on L40S.

Scores every epoch_*.pth, including those already on disk. Retries if the
eval job died without panda_plus_clean30.json. Never H200. Never scancels train.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
MIRROR = PROJECT / "outputs" / "_code_mirror"
TAG = "opt3_omar6_round2redo_tau06_benign"
CKPT_DIR = PROJECT / "outputs" / "checkpoints" / f"uni2_upernet_raw_{TAG}"
EVAL_ROOT = PROJECT / "outputs" / "pseudo_label" / "epoch_eval" / TAG
LEAK_ROOT = PROJECT / "outputs" / "docs" / "slide_groups" / "panda_plus_lookalike_gallery" / "leak_split"
STATE_PATH = PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher_r2redo_state.json"
LOG_PATH = PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher_r2redo.log"
EVAL_SCRIPT = MIRROR / "scripts" / "slurm_eval_opt3_panda_plus_r2redo.sh"
if not EVAL_SCRIPT.is_file():
    EVAL_SCRIPT = PROJECT / "scripts" / "slurm_eval_opt3_panda_plus_r2redo.sh"
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")
EVAL_JOB_NAME = "pp_r2redo_eval"
INTERVAL = int(os.environ.get("INTERVAL_SEC", "60"))
REQUIRED_SRC = (
    "evaluate.py",
    "evaluate_panda_plus_leak_split.py",
    "panda_plus_gleason_mismatch.py",
    "patch_utils.py",
    "isup_diagnostic.py",
    "train/__init__.py",
    "train/baseline_dataset.py",
    "train/model.py",
    "train/uni2_upernet.py",
    "train/lora_vit.py",
)
SRC_SEARCH = (
    MIRROR / "src",
    PROJECT / "outputs" / "_restore" / "src",
    PROJECT / "src",
)
PIN_SRC = PROJECT / "outputs" / "_restore" / "src"
MIN_SRC_BYTES = 80
MAX_INSTANT_FAILS = 3
BACKOFF_SEC = 300
INSTANT_FAIL_SEC = 30

_LOG_FP = None


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


def _usable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < MIN_SRC_BYTES:
        return False
    try:
        ast.parse(data.decode("utf-8", errors="replace"))
    except SyntaxError:
        return False
    return True


def recover_src_files() -> tuple[bool, str]:
    """Copy missing/empty/broken required files into the mirror src and pin them."""
    target = MIRROR / "src"
    target.mkdir(parents=True, exist_ok=True)
    PIN_SRC.mkdir(parents=True, exist_ok=True)
    missing = []
    notes = []
    for name in REQUIRED_SRC:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        pin = PIN_SRC / name
        if _usable(dest):
            if not _usable(pin):
                pin.parent.mkdir(parents=True, exist_ok=True)
                pin.write_bytes(dest.read_bytes())
                notes.append(f"pinned {name}")
            continue
        src = next((p / name for p in SRC_SEARCH if _usable(p / name)), None)
        if src is None:
            missing.append(name)
            continue
        dest.write_bytes(src.read_bytes())
        notes.append(f"recovered {name} from {src}")
        if not _usable(pin):
            pin.parent.mkdir(parents=True, exist_ok=True)
            pin.write_bytes(src.read_bytes())
    if missing:
        searched = ", ".join(str(p) for p in SRC_SEARCH)
        return False, f"missing {missing}; searched {searched}"
    extra = ("; " + "; ".join(notes)) if notes else ""
    return True, f"src ready at {target}{extra}"


def _last_line(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:240]
    return ""


def last_eval_error(job_id: int | None) -> str:
    if not job_id:
        return ""
    out = _last_line(PROJECT / "outputs" / "logs" / f"eval_opt3_pp_r2redo_{job_id}.out")
    err = _last_line(PROJECT / "outputs" / "logs" / f"eval_opt3_pp_r2redo_{job_id}.err")
    return " | ".join(x for x in (out, err) if x)


def load_state() -> dict:
    if STATE_PATH.is_file():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("active_job_id", None)
        data.setdefault("submitted", [])
        data.setdefault("instant_fails", 0)
        data.setdefault("backoff_until", 0.0)
        data.setdefault("last_submit_ts", 0.0)
        return data
    return {
        "active_job_id": None,
        "submitted": [],
        "instant_fails": 0,
        "backoff_until": 0.0,
        "last_submit_ts": 0.0,
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def list_epochs() -> list[tuple[int, Path]]:
    found = []
    if not CKPT_DIR.is_dir():
        return found
    for p in sorted(CKPT_DIR.glob("epoch_*_cancer_*.pth")):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def epoch_done(ep: int) -> bool:
    ep3 = f"{ep:03d}"
    leak = LEAK_ROOT / f"round2redo_ep{ep3}" / "panda_plus_leak_split.json"
    clean = EVAL_ROOT / f"ep{ep3}" / "panda_plus_clean30.json"
    isup = EVAL_ROOT / f"ep{ep3}" / "panda_plus_isup.csv"
    return leak.is_file() and leak.stat().st_size > 0 and clean.is_file() and isup.is_file()


def slurm_job_active(job_id: int | None) -> bool:
    if not job_id:
        return False
    try:
        r = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"WARN squeue failed for {job_id}: {exc}")
        return True
    state = (r.stdout or "").strip().splitlines()
    if not state:
        return False
    return state[0] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}


def any_eval_job_active() -> bool:
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "lea14"), "-n", EVAL_JOB_NAME, "-h", "-o", "%i %T"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}:
            return True
    return False


def submit(ep: int, ckpt: Path) -> int | None:
    train_log = CKPT_DIR / "training_log.csv"
    out_dir = EVAL_ROOT / f"ep{ep:03d}"
    cmd = [
        "sbatch",
        "--parsable",
        "--gres=gpu:l40s:1",
        "--exclude=esplhpc-cp087,esplhpc-cp088",
        (
            f"--export=ALL,RUN_TAG={TAG},EPOCH={ep},"
            f"OUT_DIR={out_dir},TRAIN_LOG={train_log},PANDA_PROJECT={PROJECT}"
        ),
        str(EVAL_SCRIPT),
        str(ckpt),
    ]
    log(f"SUBMIT {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        log(f"SUBMIT_FAIL rc={r.returncode} stderr={(r.stderr or '').strip()}")
        return None
    try:
        job_id = int(out.split(";")[0])
    except ValueError:
        log(f"SUBMIT_FAIL unparsable sbatch stdout={out!r}")
        return None
    log(f"SUBMIT_OK job={job_id} tag={TAG} ep={ep:03d} ckpt={ckpt.name}")
    return job_id


def tick(state: dict) -> dict:
    now = time.time()
    backoff_until = float(state.get("backoff_until") or 0.0)
    if now < backoff_until:
        log(f"BACKOFF {int(backoff_until - now)}s left (will not resubmit a broken eval)")
        return state

    active = state.get("active_job_id")
    if slurm_job_active(active) or any_eval_job_active():
        log(f"WAIT eval still queued/running job={active}")
        return state
    if active:
        err = last_eval_error(active)
        elapsed = now - float(state.get("last_submit_ts") or 0)
        log(f"PRIOR job {active} left squeue after {elapsed:.0f}s last_line={err!r}")
        pending_same = [(ep, ckpt) for ep, ckpt in list_epochs() if not epoch_done(ep)]
        died_fast = (
            "ERROR" in err
            or "missing" in err.lower()
            or "FAIL " in err
            or elapsed < INSTANT_FAIL_SEC
        )
        if died_fast and pending_same:
            state["instant_fails"] = int(state.get("instant_fails") or 0) + 1
            if state["instant_fails"] >= MAX_INSTANT_FAILS:
                state["backoff_until"] = now + BACKOFF_SEC
                log(
                    f"HALT resubmit loop after {state['instant_fails']} instant "
                    f"failures. Fix src, then watcher will retry in {BACKOFF_SEC}s. "
                    f"reason={err!r}"
                )
                state["active_job_id"] = None
                return state
        elif not died_fast:
            state["instant_fails"] = 0
        state["active_job_id"] = None

    ok, detail = recover_src_files()
    if not ok:
        state["instant_fails"] = int(state.get("instant_fails") or 0) + 1
        state["backoff_until"] = now + BACKOFF_SEC
        log(f"PREFLIGHT_FAIL {detail}; backoff {BACKOFF_SEC}s")
        return state
    if "recovered" in detail:
        log(f"PREFLIGHT {detail}")
        state["instant_fails"] = 0

    pending = [(ep, ckpt) for ep, ckpt in list_epochs() if not epoch_done(ep)]
    if not pending:
        log(f"IDLE scored={sum(1 for ep, _ in list_epochs() if epoch_done(ep))} waiting for next epoch_*.pth")
        state["instant_fails"] = 0
        return state
    ep, ckpt = pending[0]
    log(f"NEXT ep={ep:03d} remaining={len(pending)} ckpt={ckpt.name}")
    job_id = submit(ep, ckpt)
    if job_id is None:
        return state
    state["active_job_id"] = job_id
    state["last_submit_ts"] = time.time()
    state.setdefault("submitted", []).append(
        {"tag": TAG, "epoch": ep, "ckpt": str(ckpt), "job_id": job_id}
    )
    return state


def main() -> None:
    global _LOG_FP
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = LOG_PATH.open("a", encoding="utf-8")
    state = load_state()
    log(f"START tag={TAG} script={EVAL_SCRIPT} interval={INTERVAL}s")
    try:
        while True:
            state = tick(state)
            save_state(state)
            time.sleep(INTERVAL)
    finally:
        _LOG_FP.close()
        _LOG_FP = None


if __name__ == "__main__":
    main()
