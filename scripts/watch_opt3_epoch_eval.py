#!/usr/bin/env python3
"""Watch Opt3 ckpt dirs and enqueue PANDA / PANDA+ Dice+ISUP evals.

Backfill missing epochs. Several H100/A100 jobs at once (never H200).
AUTO_SUBMIT defaults on. Does not scancel H200 trains.
"""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
MIRROR = PROJECT / "outputs" / "_code_mirror"
DEFAULT_CONFIG = MIRROR / "scripts" / "epoch_eval_targets.json"
DEFAULT_LOG = PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher.log"
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")
EVAL_JOB_NAMES = {"opt3_ep_eval", "opt3_pp_eval"}
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
USER = os.environ.get("USER", "lea14")

_LOG_FP = None


def log(msg):
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = "{0} | {1}".format(ts, msg)
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


def code_file(*parts):
    mirror = MIRROR.joinpath(*parts)
    live = PROJECT.joinpath(*parts)
    return mirror if mirror.is_file() else live


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not cfg.get("targets"):
        raise SystemExit("{0} has no targets[]".format(path))
    cfg.setdefault("eval_script", str(code_file("scripts", "slurm_eval_opt3_epoch.sh")))
    cfg.setdefault(
        "state_file",
        str(PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher_state.json"),
    )
    cfg.setdefault("eval_root", str(PROJECT / "outputs" / "pseudo_label" / "epoch_eval"))
    cfg.setdefault("backfill", True)
    cfg.setdefault("max_parallel", 4)
    cfg.setdefault("gpus", ["h100", "a100"])
    return cfg


def load_state(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "baseline_epochs": {},
        "queued": [],
        "active_job_id": None,
        "submitted": [],
    }


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def eval_dir_for(cfg, tag, epoch):
    return Path(cfg["eval_root"]) / tag / "ep{0:03d}".format(epoch)


def list_epoch_files(ckpt_dir):
    found = []
    if not ckpt_dir.is_dir():
        return found
    for p in sorted(ckpt_dir.glob("epoch_*_cancer_*.pth")):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def eval_status(eval_dir):
    """Complete only when PANDA ISUP + PANDA+ Dice + PANDA+ ISUP all exist.

    Partial summaries and nonempty dirs are treated as missing so a crashed
    eval is retried. Never skip an epoch just because it was on disk at
    watcher start (no baseline_epochs gate).
    """
    summary = eval_dir / "summary.json"
    plus = eval_dir / "panda_plus_isup_summary.json"
    panda = eval_dir / "panda_isup_summary.json"
    dice = eval_dir / "panda_plus_dice_labeled.csv"
    if not (summary.is_file() and plus.is_file() and panda.is_file() and dice.is_file()):
        return "missing"
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "missing"
    if payload.get("panda_plus_isup_match") is None:
        return "missing"
    if payload.get("panda_plus_cancer_dice") is None:
        return "missing"
    if payload.get("panda_isup_match") is None:
        return "missing"
    status = str(payload.get("status", "")).lower()
    if status == "complete":
        return "complete"
    return "missing"


def queue_key(tag, epoch):
    return "{0}:ep{1:03d}".format(tag, epoch)


def list_eval_job_ids():
    try:
        r = subprocess.run(
            ["squeue", "-u", USER, "-h", "-o", "%i %j %T"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("WARN squeue failed: {0}".format(exc))
        return None
    if r.returncode != 0:
        log("WARN squeue rc={0} {1}".format(r.returncode, (r.stderr or "").strip()))
        return None
    ids = set()
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name, state = parts[1], parts[2]
        if name in EVAL_JOB_NAMES and state in ACTIVE_STATES:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                continue
    return ids


def detect_new(state, targets, cfg, active_ids):
    queued_keys = {queue_key(q["tag"], int(q["epoch"])) for q in state.get("queued", [])}
    inflight_keys = set()
    if active_ids is not None:
        for s in state.get("submitted", []):
            if s.get("job_id") in active_ids:
                inflight_keys.add(queue_key(s["tag"], int(s["epoch"])))
    now = time.time()
    last_submit = state.get("last_submit") or {}
    found = []
    for t in targets:
        tag = t["tag"]
        for ep, ckpt in list_epoch_files(Path(t["ckpt_dir"])):
            dest = eval_dir_for(cfg, tag, ep)
            key = queue_key(tag, ep)
            st = eval_status(dest)
            if st in {"complete", "partial"}:
                continue
            if key in queued_keys or key in inflight_keys:
                continue
            if now - float(last_submit.get(key, 0)) < SUBMIT_COOLDOWN_SEC:
                continue
            item = {
                "tag": tag,
                "epoch": ep,
                "ckpt": str(ckpt),
                "eval_dir": str(dest),
                "train_log": t.get("train_log", ""),
            }
            found.append(item)
            log("DETECT tag={0} ep={1:03d} ckpt={2} -> {3}".format(tag, ep, ckpt.name, dest))
    return found


PRIORITY_TAG = "opt3_omar6_round2_ep14ref"
SUBMIT_COOLDOWN_SEC = 900


def prioritize_queue(state):
    """Round 2 first (newest epoch first), then everyone else. Dedup keys."""
    queued = state.get("queued") or []
    seen = set()
    uniq = []
    for item in queued:
        key = queue_key(item["tag"], int(item["epoch"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    uniq.sort(
        key=lambda it: (
            0 if it.get("tag") == PRIORITY_TAG else 1,
            -int(it["epoch"]),
        )
    )
    state["queued"] = uniq


def submit_eval(item, cfg, gres):
    # Live scripts/ is often wiped. Never fall back to a missing live path.
    script = str(MIRROR / "scripts" / "slurm_eval_opt3_epoch.sh")
    if not Path(script).is_file():
        log("SUBMIT_FAIL missing mirror eval script {0}".format(script))
        return None
    bs = "8" if "a100" in gres else str(cfg.get("eval_bs", 16))
    cmd = [
        "sbatch",
        "--parsable",
        "--cpus-per-task=4",
        "--gres={0}".format(gres),
        "--export=ALL,RUN_TAG={0},EPOCH={1},OUT_DIR={2},TRAIN_LOG={3},EVAL_BS={4}".format(
            item["tag"],
            int(item["epoch"]),
            item["eval_dir"],
            item.get("train_log", ""),
            bs,
        ),
        script,
        item["ckpt"],
    ]
    log("SUBMIT {0}".format(" ".join(cmd)))
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        log("SUBMIT_FAIL sbatch timed out")
        return None
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        log("SUBMIT_FAIL rc={0} stderr={1}".format(r.returncode, (r.stderr or "").strip()))
        return None
    try:
        job_id = int(out.split(";")[0])
    except ValueError:
        log("SUBMIT_FAIL unparsable sbatch stdout={0!r}".format(out))
        return None
    log(
        "SUBMIT_OK job={0} gres={1} tag={2} ep={3:03d}".format(
            job_id, gres, item["tag"], int(item["epoch"])
        )
    )
    return job_id


def drain_queue(state, cfg, auto_submit, active_ids):
    queued = state.setdefault("queued", [])
    if not queued:
        return
    max_p = int(cfg.get("max_parallel", 4))
    gpus = list(cfg.get("gpus") or ["h100", "a100"])
    if active_ids is None:
        log("QUEUE hold (squeue unknown) depth={0}".format(len(queued)))
        return
    n_active = len(active_ids)
    log(
        "QUEUE active={0}/{1} depth={2} auto_submit={3}".format(
            n_active, max_p, len(queued), int(auto_submit)
        )
    )
    if not auto_submit:
        log("QUEUE hold (AUTO_SUBMIT=0) — detection only, no sbatch")
        return
    submitted_n = len(state.get("submitted", []))
    while queued and n_active < max_p:
        head = queued[0]
        gres = "gpu:{0}:1".format(gpus[submitted_n % len(gpus)])
        job_id = submit_eval(head, cfg, gres)
        if job_id is None:
            alt = gpus[(submitted_n + 1) % len(gpus)]
            alt_gres = "gpu:{0}:1".format(alt)
            if alt_gres != gres:
                job_id = submit_eval(head, cfg, alt_gres)
                if job_id is not None:
                    gres = alt_gres
        if job_id is None:
            log("QUEUE submit failed; will retry next tick")
            return
        queued.pop(0)
        n_active += 1
        submitted_n += 1
        active_ids.add(job_id)
        state["active_job_id"] = job_id
        state.setdefault("submitted", []).append(
            dict(head, job_id=job_id, gres=gres)
        )
        state.setdefault("last_submit", {})[
            queue_key(head["tag"], int(head["epoch"]))
        ] = time.time()


def tick(cfg, state, auto_submit):
    active_ids = list_eval_job_ids()
    new_items = detect_new(state, cfg["targets"], cfg, active_ids)
    for item in new_items:
        state.setdefault("queued", []).append(item)
        log(
            "ENQUEUE tag={0} ep={1:03d} queue_depth={2}".format(
                item["tag"], int(item["epoch"]), len(state["queued"])
            )
        )
    prioritize_queue(state)
    drain_queue(state, cfg, auto_submit, active_ids)
    return state


def main():
    ap = argparse.ArgumentParser(description="Per-epoch PANDA / PANDA+ eval watcher")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--interval-sec", type=int, default=int(os.environ.get("INTERVAL_SEC", "60")))
    ap.add_argument(
        "--auto-submit",
        action="store_true",
        default=os.environ.get("AUTO_SUBMIT", "1").strip().lower() in {"1", "true", "yes"},
    )
    ap.add_argument("--no-auto-submit", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--log-file",
        type=Path,
        default=Path(os.environ.get("WATCH_LOG", str(DEFAULT_LOG))),
    )
    args = ap.parse_args()
    auto_submit = False if args.no_auto_submit else args.auto_submit

    global _LOG_FP
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = args.log_file.open("a", encoding="utf-8")
    cfg = load_config(args.config)
    state_path = Path(cfg["state_file"])
    state = load_state(state_path)
    log(
        "START config={0} targets={1} auto_submit={2} interval={3}s "
        "backfill={4} max_parallel={5} gpus={6} log={7}".format(
            args.config,
            [t["tag"] for t in cfg["targets"]],
            int(auto_submit),
            args.interval_sec,
            int(bool(cfg.get("backfill", True))),
            cfg.get("max_parallel"),
            cfg.get("gpus"),
            args.log_file,
        )
    )
    try:
        while True:
            cfg = load_config(args.config)
            state = tick(cfg, state, auto_submit)
            save_state(state_path, state)
            if args.once:
                log("ONCE done")
                return
            time.sleep(args.interval_sec)
    finally:
        _LOG_FP.close()


if __name__ == "__main__":
    main()
