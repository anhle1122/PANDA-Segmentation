#!/usr/bin/env python3
"""Self-heal Round-2-redo train + keep PANDA+ watcher alive.

Watches the opt3_r2redo train job. On failure:
  1. Read train .out/.err and classify the error
  2. Restore missing code/scripts from git + _code_mirror pins
  3. Resubmit train (RESUME_OK=1 if any epoch_*.pth exists)
  4. Ensure the PANDA+ watcher job is running

Never scancels a healthy RUNNING/PENDING train.
Never touches Round 4 / other tags.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
MIRROR = PROJECT / "outputs" / "_code_mirror"
RESTORE = PROJECT / "outputs" / "_restore"
SCRIPTS = PROJECT / "scripts"
MIRROR_SCRIPTS = MIRROR / "scripts"
TAG = "opt3_omar6_round2redo_tau06_benign"
TRAIN_JOB_NAME = "opt3_r2redo"
TRAIN_JOB_NAMES = "opt3_r2redo,opt3_r2redo_h200x2,opt3_r2redo_h200x4,opt3_r2redo_h100,opt3_r2redo_l40s"
H200_2X_NAME = "opt3_r2redo_h200x2"
H200_4X_NAME = "opt3_r2redo_h200x4"
WATCH_JOB_NAME = "opt3_r2redo_pp_watch"
TRAIN_SCRIPT = SCRIPTS / "slurm_train_opt3_round2redo_corrected.sh"
WATCH_SCRIPT = SCRIPTS / "slurm_watch_r2redo_panda_plus.sh"
EVAL_SCRIPT = SCRIPTS / "slurm_eval_opt3_panda_plus_r2redo.sh"
PP_WATCH_PY = SCRIPTS / "watch_r2redo_panda_plus.py"
CKPT_DIR = PROJECT / "outputs" / "checkpoints" / f"uni2_upernet_raw_{TAG}"
CORRECTED_DIR = (
    PROJECT / "outputs" / "pseudo_label" / "corrections_opt3_omar6_locked_r2_ep014_tau06_benign"
)
STATE_PATH = PROJECT / "outputs" / "pseudo_label" / "r2redo_train_watchdog_state.json"
LOG_PATH = PROJECT / "outputs" / "pseudo_label" / "r2redo_train_watchdog.log"
INTERVAL = int(os.environ.get("INTERVAL_SEC", "90"))
MAX_RESUBMITS_PER_HOUR = 3
BACKOFF_SEC = 600
INSTANT_FAIL_SEC = 120

# Files that must exist for train to start. Restored from git HEAD if missing.
REQUIRED_SRC = (
    "train_uni2_opt3_slidebag.py",
    "train_baseline.py",
    "patch_utils.py",
    "train/__init__.py",
    "train/baseline_dataset.py",
    "train/corrected_label_dataset.py",
    "train/class_weights.py",
    "train/grade_head.py",
    "train/losses.py",
    "train/metrics.py",
    "train/slide_bag_dataset.py",
    "train/uni2_upernet.py",
    "train/lora_vit.py",
    "train/model.py",
    "train/pseudo_label_rules.py",
)

PIN_SCRIPTS = (
    "slurm_train_opt3_round2redo_corrected.sh",
    "slurm_eval_opt3_panda_plus_r2redo.sh",
    "slurm_watch_r2redo_panda_plus.sh",
    "watch_r2redo_panda_plus.py",
    "slurm_watch_r2redo_train_watchdog.sh",
    "watch_r2redo_train_watchdog.py",
    "hpc_use_code.sh",
)

HPC_USE_CODE = """# Prefer code mirror for training/eval; fall back to _restore.
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/src/train_uni2_opt3_slidebag.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
elif [[ -f "${PANDA_PROJECT}/outputs/_restore/src/train_uni2_opt3_slidebag.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_restore/src"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
fi
export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
echo "PANDA_CODE_SRC=${PANDA_CODE_SRC}"
"""

_LOG_FP = None


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def load_state() -> dict:
    if STATE_PATH.is_file():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        data = {}
    data.setdefault("train_job_id", None)
    data.setdefault("watch_job_id", None)
    data.setdefault("h200x2_job_id", None)
    data.setdefault("h200x4_job_id", None)
    data.setdefault("resubmit_times", [])
    data.setdefault("backoff_until", 0.0)
    data.setdefault("last_fail_reason", "")
    data.setdefault("n_heal", 0)
    return data


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def job_state(job_id: int | None) -> str | None:
    if not job_id:
        return None
    try:
        r = run(["squeue", "-j", str(job_id), "-h", "-o", "%T"], timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    lines = (r.stdout or "").strip().splitlines()
    return lines[0] if lines else None


def jobs_by_name(name: str) -> list[tuple[int, str]]:
    try:
        r = run(
            ["squeue", "-u", os.environ.get("USER", "lea14"), "-n", name, "-h", "-o", "%i %T"],
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out.append((int(parts[0]), parts[1]))
            except ValueError:
                continue
    return out


def sacct_state(job_id: int) -> tuple[str, str]:
    r = run(
        [
            "sacct",
            "-j",
            str(job_id),
            "-n",
            "-X",
            "-P",
            "--format=State,ExitCode,Elapsed",
        ],
        timeout=30,
    )
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return "UNKNOWN", ""
    parts = line[0].split("|")
    state = parts[0].split()[0] if parts else "UNKNOWN"
    detail = "|".join(parts)
    return state, detail


def last_log_snippet(job_id: int, n: int = 40) -> str:
    chunks = []
    for suf in (".out", ".err"):
        path = PROJECT / "outputs" / "logs" / f"train_opt3_r2redo_{job_id}{suf}"
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        tail = "\n".join(lines[-n:])
        if tail.strip():
            chunks.append(f"===== {path.name} =====\n{tail}")
    return "\n".join(chunks)


def classify_failure(text: str) -> str:
    t = text.lower()
    if "missing" in t and "train_uni2_opt3_slidebag.py" in t:
        return "missing_train_script"
    if "missing hpc_use_code.sh" in t or "missing" in t and "hpc_use_code" in t:
        return "missing_hpc_use_code"
    if "missing corrected dir" in t or "incomplete (no balance_report" in t:
        return "missing_corrected"
    if "already has weights" in t:
        return "ckpt_exists_need_resume"
    if "nccl" in t or "libnccl" in t:
        return "nccl"
    if "cuda" in t and ("out of memory" in t or "oom" in t):
        return "oom"
    if "error: missing" in t:
        return "missing_file"
    if "traceback" in t:
        return "python_traceback"
    return "unknown"


def git_show(rel: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = run(["git", "-C", str(PROJECT), "show", f"HEAD:src/{rel}"], timeout=60)
    if r.returncode != 0 or not (r.stdout or "").strip():
        return False
    dest.write_text(r.stdout, encoding="utf-8")
    return dest.is_file() and dest.stat().st_size > 50


def ensure_file(rel: str) -> bool:
    """Ensure src file exists in mirror and restore. Prefer existing, else git."""
    targets = [MIRROR / "src" / rel, RESTORE / "src" / rel]
    ok_any = False
    for t in targets:
        if t.is_file() and t.stat().st_size > 50:
            ok_any = True
    if ok_any:
        # pin both ways
        src = next(t for t in targets if t.is_file() and t.stat().st_size > 50)
        for t in targets:
            if t != src and (not t.is_file() or t.stat().st_size < 50):
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, t)
        return True
    # restore from git into both
    tmp = MIRROR / "src" / rel
    if git_show(rel, tmp):
        pin = RESTORE / "src" / rel
        pin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp, pin)
        log(f"HEAL restored src/{rel} from git")
        return True
    return False


def ensure_hpc_use_code() -> None:
    path = MIRROR_SCRIPTS / "hpc_use_code.sh"
    MIRROR_SCRIPTS.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size < 40:
        path.write_text(HPC_USE_CODE, encoding="utf-8")
        log("HEAL wrote hpc_use_code.sh")
    path.chmod(0o755)


def ensure_scripts() -> list[str]:
    """Copy pinned scripts from mirror <-> scripts/."""
    notes = []
    MIRROR_SCRIPTS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    ensure_hpc_use_code()
    for name in PIN_SCRIPTS:
        if name == "hpc_use_code.sh":
            continue
        live = SCRIPTS / name
        pin = MIRROR_SCRIPTS / name
        if live.is_file() and live.stat().st_size > 50:
            if not pin.is_file() or pin.stat().st_size < 50:
                shutil.copy2(live, pin)
                notes.append(f"pinned {name}")
        elif pin.is_file() and pin.stat().st_size > 50:
            shutil.copy2(pin, live)
            live.chmod(0o755)
            notes.append(f"restored {name} from mirror")
        else:
            notes.append(f"MISSING_SCRIPT {name}")
    return notes


def heal_code() -> tuple[bool, str]:
    missing = []
    for rel in REQUIRED_SRC:
        if not ensure_file(rel):
            missing.append(rel)
    notes = ensure_scripts()
    if missing:
        return False, f"still missing {missing}; scripts={notes}"
    # corrected pack must exist (do not rebuild here — too expensive)
    if not (CORRECTED_DIR / "balance_report.json").is_file():
        return False, f"corrected pack incomplete: {CORRECTED_DIR}"
    return True, "ok; " + (", ".join(notes) if notes else "files present")


def can_resubmit(state: dict) -> bool:
    now = time.time()
    if now < float(state.get("backoff_until") or 0):
        return False
    recent = [t for t in state.get("resubmit_times", []) if now - t < 3600]
    state["resubmit_times"] = recent
    return len(recent) < MAX_RESUBMITS_PER_HOUR


def _has_ckpt() -> bool:
    return bool(list(CKPT_DIR.glob("epoch_*.pth"))) or (CKPT_DIR / "latest.pth").is_file()


def submit_named_train(job_name: str, extra: list[str] | None = None) -> int | None:
    if not TRAIN_SCRIPT.is_file():
        log(f"SUBMIT_FAIL missing {TRAIN_SCRIPT}")
        return None
    env_export = f"ALL,PANDA_PROJECT={PROJECT},RESUME_OK=1"
    if _has_ckpt():
        log(f"RESUME_OK=1 (existing ckpts found) job_name={job_name}")
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        *(extra or []),
        f"--export={env_export}",
        str(TRAIN_SCRIPT),
    ]
    log(f"SUBMIT_TRAIN {' '.join(cmd)}")
    r = run(cmd, timeout=60)
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        log(f"SUBMIT_FAIL rc={r.returncode} stderr={(r.stderr or '').strip()}")
        return None
    try:
        job_id = int(out.split(";")[0])
    except ValueError:
        log(f"SUBMIT_FAIL unparsable {out!r}")
        return None
    log(f"SUBMIT_OK train job={job_id} name={job_name}")
    return job_id


def submit_train(state: dict) -> int | None:
    ok, detail = heal_code()
    if not ok:
        log(f"HEAL_FAIL before submit: {detail}")
        state["backoff_until"] = time.time() + BACKOFF_SEC
        return None
    log(f"HEAL_OK {detail}")
    job_id = submit_named_train(H200_2X_NAME, ["--gres=gpu:h200:2"])
    if job_id:
        state["resubmit_times"].append(time.time())
        state["n_heal"] = int(state.get("n_heal") or 0) + 1
    return job_id


def _gres_count(gres: str, kind: str) -> int:
    m = re.search(rf"gpu:{re.escape(kind)}:(\d+)", gres or "")
    return int(m.group(1)) if m else 0


def h200_idle_nodes() -> list[tuple[str, int, int, int, str]]:
    """Return (host, configured, used, idle, state) for H200 nodes. Skip 087/088."""
    try:
        r = run(
            [
                "sinfo",
                "-N",
                "-h",
                "-p",
                "gpu",
                "-O",
                "NodeHost:20,Gres:40,GresUsed:40,StateCompact:10",
            ],
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        host, gres, gres_used, state = parts[0], parts[1], parts[2], parts[3]
        if "h200" not in gres:
            continue
        if host in {"esplhpc-cp087", "esplhpc-cp088"}:
            continue
        cfg = _gres_count(gres, "h200")
        used = _gres_count(gres_used, "h200")
        idle = max(0, cfg - used)
        out.append((host, cfg, used, idle, state))
    return out


def maybe_promote_to_4x() -> None:
    """If 2xH200 is running and 2 more H200 are free on that 4-GPU node, release
    2x so the already-pending 4x can take the full node. Never cancel the 4x wait.
    If some other node already has 4 idle, leave 2x alone — 4x will start itself.
    """
    two = jobs_by_name(H200_2X_NAME)
    four = jobs_by_name(H200_4X_NAME)
    two_run = [j for j, st in two if st in {"RUNNING", "CONFIGURING"}]
    four_run = any(st in {"RUNNING", "CONFIGURING", "COMPLETING"} for _, st in four)
    four_wait = any(active_states(st) and st not in {"RUNNING", "CONFIGURING", "COMPLETING"} for _, st in four)
    if four_run or not two_run or not four_wait:
        return
    nodes = h200_idle_nodes()
    if not nodes:
        return
    if any(idle >= 4 for _, _, _, idle, _ in nodes):
        log("H200_PROMOTE wait — a node has 4 idle H200; 4x will start, keep 2x until then")
        return
    r = run(["squeue", "-j", str(two_run[0]), "-h", "-o", "%N"], timeout=30)
    host = (r.stdout or "").strip().split()[0] if (r.stdout or "").strip() else ""
    for n_host, cfg, used, idle, _state in nodes:
        if n_host == host and cfg >= 4 and idle >= 2:
            log(
                f"H200_PROMOTE {host} cfg={cfg} used={used} idle={idle} "
                f"— scancel 2x job={two_run[0]} so pending 4x can take 4"
            )
            run(["scancel", str(two_run[0])], timeout=30)
            return
    idle_s = ",".join(f"{h}:{idle}" for h, _, _, idle, _ in nodes)
    log(f"H200_PROMOTE hold 2x job={two_run[0]} on {host or '?'} idle=[{idle_s}]")


def ensure_h200_queue(state: dict) -> None:
    """Keep 2x and 4x H200 queued. Never drop a pending H200 while waiting.

    L40S/2x running is not a reason to skip this. 4x stays pending until a
    full node frees; 2x stays pending until two cards free. 4x RUNNING means
    the switch is done — do not submit more train jobs.
    """
    if (CKPT_DIR / "TRAINING_COMPLETE.txt").is_file():
        log("TRAIN_COMPLETE skip h200 queue")
        return
    four = jobs_by_name(H200_4X_NAME)
    two = jobs_by_name(H200_2X_NAME)
    four_run = any(st in {"RUNNING", "CONFIGURING", "COMPLETING"} for _, st in four)
    four_live = any(active_states(st) for _, st in four)
    two_live = any(active_states(st) for _, st in two)
    if four_run:
        log(f"H200x4_OK job={four[0][0]} — switch done, keep 4x only")
        return
    if not four_live:
        jid = submit_named_train(
            H200_4X_NAME,
            ["--gres=gpu:h200:4", "--cpus-per-task=16", "--mem=192G"],
        )
        if jid:
            state["h200x4_job_id"] = jid
            state["resubmit_times"].append(time.time())
    else:
        state["h200x4_job_id"] = four[0][0]
        log(f"H200x4_QUEUED job={four[0][0]} state={four[0][1]}")
    if not two_live:
        jid = submit_named_train(H200_2X_NAME, ["--gres=gpu:h200:2"])
        if jid:
            state["h200x2_job_id"] = jid
            state["resubmit_times"].append(time.time())
    else:
        state["h200x2_job_id"] = two[0][0]
        log(f"H200x2_QUEUED job={two[0][0]} state={two[0][1]}")


def submit_pp_watcher() -> int | None:
    # recreate watch script if missing
    if not WATCH_SCRIPT.is_file():
        pin = MIRROR_SCRIPTS / "slurm_watch_r2redo_panda_plus.sh"
        if pin.is_file():
            shutil.copy2(pin, WATCH_SCRIPT)
            WATCH_SCRIPT.chmod(0o755)
        else:
            WATCH_SCRIPT.write_text(
                """#!/usr/bin/env bash
#SBATCH --job-name=opt3_r2redo_pp_watch
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_pp_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_pp_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export INTERVAL_SEC="${INTERVAL_SEC:-60}"
python -u "${PANDA_PROJECT}/scripts/watch_r2redo_panda_plus.py"
""",
                encoding="utf-8",
            )
            WATCH_SCRIPT.chmod(0o755)
    if not PP_WATCH_PY.is_file():
        pin = MIRROR_SCRIPTS / "watch_r2redo_panda_plus.py"
        if pin.is_file():
            shutil.copy2(pin, PP_WATCH_PY)
            PP_WATCH_PY.chmod(0o755)
    r = run(["sbatch", "--parsable", str(WATCH_SCRIPT)], timeout=60)
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        log(f"WATCH_SUBMIT_FAIL {(r.stderr or '').strip()}")
        return None
    try:
        job_id = int(out.split(";")[0])
    except ValueError:
        return None
    log(f"WATCH_SUBMIT_OK job={job_id}")
    return job_id


def active_states(st: str | None) -> bool:
    return st in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}


def tick(state: dict) -> dict:
    now = time.time()

    # Keep PANDA+ watcher alive
    watchers = jobs_by_name(WATCH_JOB_NAME)
    if not any(active_states(st) for _, st in watchers):
        if can_resubmit(state):
            wid = submit_pp_watcher()
            if wid:
                state["watch_job_id"] = wid
        else:
            log("WATCH_DEAD but backoff/rate-limit — skip resubmit")
    else:
        state["watch_job_id"] = watchers[0][0]

    # Keep 2x/4x H200 queued even while L40S or 2x is already running.
    # Do not treat "some stopgap is live" as a reason to skip the H200 wait.
    if can_resubmit(state) or jobs_by_name(H200_2X_NAME) or jobs_by_name(H200_4X_NAME):
        ensure_h200_queue(state)
    else:
        log("H200_QUEUE skip — backoff/rate-limit and no H200 jobs in queue")
    maybe_promote_to_4x()

    # Train: any sibling still counts as the run being alive (do not cold-resubmit).
    trains = jobs_by_name(TRAIN_JOB_NAMES)
    live = [(jid, st) for jid, st in trains if active_states(st)]
    if live:
        state["train_job_id"] = live[0][0]
        log(f"TRAIN_OK job={live[0][0]} state={live[0][1]}")
        return state

    # No live train — diagnose last known job or recent failed
    job_id = state.get("train_job_id")
    if job_id:
        st, detail = sacct_state(int(job_id))
        snippet = last_log_snippet(int(job_id))
        reason = classify_failure(snippet)
        state["last_fail_reason"] = reason
        log(f"TRAIN_GONE job={job_id} sacct={detail} class={reason}")
        if snippet:
            # keep log short in state file; full text only in watchdog log
            for line in snippet.splitlines()[-15:]:
                log(f"  LOG {line[:200]}")
    else:
        # discover latest failed from sacct by name
        r = run(
            [
                "sacct",
                "-u",
                os.environ.get("USER", "lea14"),
                "-n",
                "-X",
                "-P",
                "--format=JobID,JobName,State,End",
                "--name",
                TRAIN_JOB_NAME,
                "-S",
                "now-7days",
            ],
            timeout=60,
        )
        failed = []
        for line in (r.stdout or "").splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and parts[2].startswith("FAILED"):
                try:
                    failed.append(int(parts[0]))
                except ValueError:
                    pass
        if failed:
            job_id = failed[-1]
            state["train_job_id"] = job_id
            snippet = last_log_snippet(job_id)
            reason = classify_failure(snippet)
            state["last_fail_reason"] = reason
            log(f"TRAIN_FOUND_FAILED job={job_id} class={reason}")
        else:
            reason = "no_train_job"
            state["last_fail_reason"] = reason
            log("TRAIN_MISSING no active/failed job found — will submit fresh")

    if not can_resubmit(state):
        left = max(0, int(float(state.get("backoff_until") or 0) - now))
        log(f"RATE_LIMIT/BACKOFF left={left}s recent={len(state.get('resubmit_times', []))}")
        return state

    # Special case: ckpt exists without RESUME_OK
    if state.get("last_fail_reason") == "ckpt_exists_need_resume":
        log("HEAL will resubmit with RESUME_OK=1")

    if state.get("last_fail_reason") == "missing_corrected":
        log("FATAL corrected pack missing — will not spin forever")
        state["backoff_until"] = now + 3600
        return state

    new_id = submit_train(state)
    if new_id:
        state["train_job_id"] = new_id
    else:
        state["backoff_until"] = now + BACKOFF_SEC
        log(f"SUBMIT failed — backoff {BACKOFF_SEC}s")
    return state


def main() -> None:
    global _LOG_FP
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = LOG_PATH.open("a", encoding="utf-8")
    state = load_state()
    # seed current pending train if any
    live = jobs_by_name(TRAIN_JOB_NAMES)
    if live:
        state["train_job_id"] = live[0][0]
    log(f"START tag={TAG} interval={INTERVAL}s train_job={state.get('train_job_id')}")
    ok, detail = heal_code()
    log(f"BOOT_HEAL {'OK' if ok else 'FAIL'}: {detail}")
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
