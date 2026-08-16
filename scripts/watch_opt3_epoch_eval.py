#!/usr/bin/env python3
"""Watch Opt3 ckpt dirs and enqueue PANDA / PANDA+ Dice+ISUP evals.

New-epochs-only (no backfill). One L40S queue. AUTO_SUBMIT defaults on.
Does not submit teacher packs and does not scancel H200 trains.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
DEFAULT_CONFIG = PROJECT / "scripts" / "epoch_eval_targets.json"
DEFAULT_LOG = PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher.log"
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")

_LOG_FP = None


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not cfg.get("targets"):
        raise SystemExit(f"{path} has no targets[]")
    cfg.setdefault("eval_script", str(PROJECT / "scripts" / "slurm_eval_opt3_epoch.sh"))
    cfg.setdefault(
        "state_file",
        str(PROJECT / "outputs" / "pseudo_label" / "epoch_eval_watcher_state.json"),
    )
    cfg.setdefault("eval_root", str(PROJECT / "outputs" / "pseudo_label" / "epoch_eval"))
    return cfg


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "baseline_epochs": {},
        "queued": [],
        "active_job_id": None,
        "submitted": [],
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def eval_dir_for(cfg: dict, tag: str, epoch: int) -> Path:
    return Path(cfg["eval_root"]) / tag / f"ep{epoch:03d}"


def list_epoch_files(ckpt_dir: Path) -> list[tuple[int, Path]]:
    found = []
    if not ckpt_dir.is_dir():
        return found
    for p in sorted(ckpt_dir.glob("epoch_*_cancer_*.pth")):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def eval_status(eval_dir: Path) -> str:
    summary = eval_dir / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "in_flight"
        status = str(payload.get("status", "")).lower()
        if status in {"complete", "partial"}:
            return status
        return "in_flight"
    if eval_dir.is_dir() and any(eval_dir.iterdir()):
        return "in_flight"
    return "missing"


def queue_key(tag: str, epoch: int) -> str:
    return f"{tag}:ep{epoch:03d}"


def ensure_baselines(state: dict, targets: list[dict]) -> None:
    baselines: dict = state.setdefault("baseline_epochs", {})
    for t in targets:
        tag = t["tag"]
        if tag in baselines:
            continue
        eps = [ep for ep, _ in list_epoch_files(Path(t["ckpt_dir"]))]
        baselines[tag] = eps
        log(f"BASELINE tag={tag} epochs={eps or '[]'} (will not backfill these)")


def detect_new(state: dict, targets: list[dict], cfg: dict) -> list[dict]:
    queued_keys = {queue_key(q["tag"], int(q["epoch"])) for q in state.get("queued", [])}
    submitted_keys = {queue_key(s["tag"], int(s["epoch"])) for s in state.get("submitted", [])}
    found: list[dict] = []
    for t in targets:
        tag = t["tag"]
        baseline = set(int(x) for x in state.get("baseline_epochs", {}).get(tag, []))
        for ep, ckpt in list_epoch_files(Path(t["ckpt_dir"])):
            dest = eval_dir_for(cfg, tag, ep)
            key = queue_key(tag, ep)
            st = eval_status(dest)
            if ep in baseline:
                continue
            if st in {"complete", "partial"}:
                continue
            if st == "in_flight":
                log(f"SKIP tag={tag} ep={ep:03d} reason=eval_in_flight {dest}")
                continue
            if key in queued_keys or key in submitted_keys:
                continue
            item = {
                "tag": tag,
                "epoch": ep,
                "ckpt": str(ckpt),
                "eval_dir": str(dest),
                "train_log": t.get("train_log", ""),
            }
            found.append(item)
            log(f"DETECT tag={tag} ep={ep:03d} ckpt={ckpt.name} -> {dest}")
    return found


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


def submit_eval(item: dict, cfg: dict, *, after_job: int | None) -> int | None:
    script = cfg.get("eval_script")
    cmd = [
        "sbatch",
        "--parsable",
        "--gres=gpu:l40s:1",
        f"--export=ALL,RUN_TAG={item['tag']},EPOCH={int(item['epoch'])},"
        f"OUT_DIR={item['eval_dir']},TRAIN_LOG={item.get('train_log', '')}",
    ]
    if after_job:
        cmd.append(f"--dependency=afterany:{after_job}")
    cmd.extend([script, item["ckpt"]])
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
    log(f"SUBMIT_OK job={job_id} tag={item['tag']} ep={int(item['epoch']):03d}")
    return job_id


def drain_queue(state: dict, cfg: dict, *, auto_submit: bool) -> None:
    queued = state.setdefault("queued", [])
    if not queued:
        return
    active = state.get("active_job_id")
    if slurm_job_active(active):
        log(f"QUEUE wait active_job={active} depth={len(queued)}")
        return
    if active:
        log(f"QUEUE prior job {active} no longer in squeue; releasing slot")
        state["active_job_id"] = None
    head = queued[0]
    log(
        f"QUEUE head tag={head['tag']} ep={int(head['epoch']):03d} "
        f"auto_submit={int(auto_submit)} depth={len(queued)}"
    )
    if not auto_submit:
        log("QUEUE hold (AUTO_SUBMIT=0) — detection only, no sbatch")
        return
    prev = None
    if state.get("submitted"):
        prev = state["submitted"][-1].get("job_id")
    job_id = submit_eval(head, cfg, after_job=prev if slurm_job_active(prev) else None)
    if job_id is None:
        return
    queued.pop(0)
    state["active_job_id"] = job_id
    state.setdefault("submitted", []).append({**head, "job_id": job_id})


def tick(cfg: dict, state: dict, *, auto_submit: bool) -> dict:
    targets = cfg["targets"]
    ensure_baselines(state, targets)
    new_items = detect_new(state, targets, cfg)
    for item in new_items:
        state.setdefault("queued", []).append(item)
        log(f"ENQUEUE tag={item['tag']} ep={int(item['epoch']):03d} queue_depth={len(state['queued'])}")
    drain_queue(state, cfg, auto_submit=auto_submit)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-epoch PANDA / PANDA+ eval watcher")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--interval-sec", type=int, default=int(os.environ.get("INTERVAL_SEC", "60")))
    ap.add_argument(
        "--auto-submit",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("AUTO_SUBMIT", "1").strip().lower() in {"1", "true", "yes"},
    )
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log-file", type=Path, default=Path(os.environ.get("WATCH_LOG", str(DEFAULT_LOG))))
    args = ap.parse_args()

    global _LOG_FP
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = args.log_file.open("a", encoding="utf-8")
    cfg = load_config(args.config)
    state_path = Path(cfg["state_file"])
    state = load_state(state_path)
    log(
        f"START config={args.config} targets={[t['tag'] for t in cfg['targets']]} "
        f"auto_submit={int(args.auto_submit)} interval={args.interval_sec}s "
        f"log={args.log_file}"
    )
    try:
        while True:
            state = tick(cfg, state, auto_submit=args.auto_submit)
            save_state(state_path, state)
            if args.once:
                log("ONCE done")
                return
            time.sleep(args.interval_sec)
    finally:
        _LOG_FP.close()
        _LOG_FP = None


if __name__ == "__main__":
    main()
