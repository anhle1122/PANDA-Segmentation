#!/usr/bin/env python3
"""Watch one or more Opt3 ckpt dirs and enqueue teacher-pack caches.

New-epochs-only: the first time a tag is seen, epochs already on disk are the
baseline and are never backfilled. One L40S queue across all tags.
AUTO_SUBMIT defaults off (detect/enqueue only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
DEFAULT_CONFIG = PROJECT / "scripts" / "teacher_watch_targets.json"
CACHE_SCRIPT = PROJECT / "scripts" / "slurm_cache_teacher_pack.sh"
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{ts} | {msg}", flush=True)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not cfg.get("targets"):
        raise SystemExit(f"{path} has no targets[]")
    cfg.setdefault("pack_root", str(PROJECT / "outputs" / "pseudo_label"))
    cfg.setdefault("cache_script", str(CACHE_SCRIPT))
    cfg.setdefault("state_file", str(PROJECT / "outputs" / "pseudo_label" / "watcher_state.json"))
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


def pack_dir_for(target: dict, epoch: int) -> Path:
    root = Path(target.get("teacher_pack_dir") or target.get("pack_root"))
    return root / f"teacher_{target['tag']}_ep{epoch:03d}"


def list_epoch_files(ckpt_dir: Path) -> list[tuple[int, Path]]:
    found = []
    if not ckpt_dir.is_dir():
        return found
    for p in sorted(ckpt_dir.glob("epoch_*_cancer_*.pth")):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def pack_status(pack_dir: Path) -> str:
    """complete | in_flight | missing"""
    cfg = pack_dir / "pack_config.json"
    if cfg.is_file():
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "in_flight"
        status = str(payload.get("status", "")).lower()
        if status == "complete":
            return "complete"
        return "in_flight"
    if pack_dir.is_dir() and any(pack_dir.glob("*_srcpred.h5")):
        return "in_flight"
    return "missing"


def queue_key(tag: str, epoch: int) -> str:
    return f"{tag}:ep{epoch:03d}"


def ensure_baselines(state: dict, targets: list[dict]) -> None:
    """First sight of a tag: freeze current on-disk epochs so they are not backfilled."""
    baselines: dict = state.setdefault("baseline_epochs", {})
    for t in targets:
        tag = t["tag"]
        if tag in baselines:
            continue
        eps = [ep for ep, _ in list_epoch_files(Path(t["ckpt_dir"]))]
        baselines[tag] = eps
        log(f"BASELINE tag={tag} epochs={eps or '[]'} (will not backfill these)")


def detect_new(state: dict, targets: list[dict]) -> list[dict]:
    """Epochs that appeared after baseline, have no pack, and are not queued."""
    queued_keys = {queue_key(q["tag"], int(q["epoch"])) for q in state.get("queued", [])}
    submitted_keys = {queue_key(s["tag"], int(s["epoch"])) for s in state.get("submitted", [])}
    found: list[dict] = []
    for t in targets:
        tag = t["tag"]
        baseline = set(int(x) for x in state.get("baseline_epochs", {}).get(tag, []))
        for ep, ckpt in list_epoch_files(Path(t["ckpt_dir"])):
            dest = pack_dir_for(t, ep)
            key = queue_key(tag, ep)
            st = pack_status(dest)
            if ep in baseline:
                continue
            if st == "complete":
                continue
            if st == "in_flight":
                log(f"SKIP tag={tag} ep={ep:03d} reason=pack_in_flight {dest}")
                continue
            if key in queued_keys or key in submitted_keys:
                continue
            item = {
                "tag": tag,
                "epoch": ep,
                "ckpt": str(ckpt),
                "pack_dir": str(dest),
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


def submit_cache(item: dict, cfg: dict, *, after_job: int | None) -> int | None:
    script = cfg.get("cache_script", str(CACHE_SCRIPT))
    cmd = [
        "sbatch",
        "--parsable",
        "--gres=gpu:l40s:1",
        f"--export=ALL,RUN_TAG={item['tag']},OUT_DIR={item['pack_dir']}",
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
    job_id = submit_cache(head, cfg, after_job=prev if slurm_job_active(prev) else None)
    if job_id is None:
        return
    queued.pop(0)
    state["active_job_id"] = job_id
    state.setdefault("submitted", []).append({**head, "job_id": job_id})


def tick(cfg: dict, state: dict, *, auto_submit: bool) -> dict:
    targets = cfg["targets"]
    ensure_baselines(state, targets)
    for item in detect_new(state, targets):
        state.setdefault("queued", []).append(item)
        log(f"ENQUEUE tag={item['tag']} ep={int(item['epoch']):03d} queue_depth={len(state['queued'])}")
    drain_queue(state, cfg, auto_submit=auto_submit)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-tag teacher-pack watcher")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--interval-sec", type=int, default=int(os.environ.get("INTERVAL_SEC", "60")))
    ap.add_argument(
        "--auto-submit",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("AUTO_SUBMIT", "0").strip().lower() in {"1", "true", "yes"},
    )
    ap.add_argument("--once", action="store_true", help="One detect/enqueue pass then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    state_path = Path(cfg["state_file"])
    state = load_state(state_path)
    log(
        f"START config={args.config} targets={[t['tag'] for t in cfg['targets']]} "
        f"auto_submit={int(args.auto_submit)} interval={args.interval_sec}s"
    )
    while True:
        state = tick(cfg, state, auto_submit=args.auto_submit)
        save_state(state_path, state)
        if args.once:
            log("ONCE done")
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
