#!/usr/bin/env python3
"""Watch Stage-B SICAP full-train ckpt dir. No Val fold exists for this tag:
per-epoch PANDA+ clean-30 is scored as a drift curve only (never selection).
Official SICAPv2 Test is scored exactly ONCE, on the LAST epoch's checkpoint
(epoch E — the frozen recipe's rule, not a val-picked middle epoch)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
TAG = os.environ.get("RUN_TAG", "").strip()
if not TAG:
    raise SystemExit("ERROR: set RUN_TAG")
LEAK_PREFIX = os.environ.get("LEAK_PREFIX", f"{TAG}_stageB").strip()
FINAL_EPOCH = int(os.environ.get("FINAL_EPOCH", "0"))  # 0 = infer from TRAINING_COMPLETE
CKPT_DIR = PROJECT / "outputs" / "checkpoints" / f"uni2_upernet_raw_{TAG}"
EVAL_ROOT = PROJECT / "outputs" / "pseudo_label" / "epoch_eval" / TAG
LEAK_ROOT = (
    PROJECT / "outputs" / "docs" / "slide_groups" / "panda_plus_lookalike_gallery" / "leak_split"
)
SICAP_EVAL_ROOT = PROJECT / "outputs" / "evaluation" / "sicapv2"
STATE_PATH = PROJECT / "outputs" / "pseudo_label" / f"epoch_eval_watcher_{LEAK_PREFIX}_state.json"
LOG_PATH = PROJECT / "outputs" / "pseudo_label" / f"epoch_eval_watcher_{LEAK_PREFIX}.log"
PP_SCRIPT = PROJECT / "scripts" / "slurm_eval_opt3_panda_plus_r2redo.sh"
SICAP_SCRIPT = PROJECT / "scripts" / "slurm_eval_sicapv2.sh"
EPOCH_RE = re.compile(r"epoch_(\d+)_cancer_")
INTERVAL = int(os.environ.get("INTERVAL_SEC", "60"))
EVAL_BS = os.environ.get("EVAL_BS", "2")
EXCLUDE = os.environ.get(
    "EVAL_EXCLUDE",
    "esplhpc-cp075,esplhpc-cp076,esplhpc-cp087,esplhpc-cp088,esplhpc-cp097",
)


def log(msg: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"submitted_pp": [], "submitted_official": False, "active_pp": None, "active_official": None}


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


def job_active(job_id) -> bool:
    if not job_id:
        return False
    r = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"], capture_output=True, text=True, timeout=30, check=False
    )
    lines = (r.stdout or "").strip().splitlines()
    return bool(lines) and lines[0] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}


def pp_done(ep: int) -> bool:
    ep3 = f"{ep:03d}"
    leak = LEAK_ROOT / f"{LEAK_PREFIX}_ep{ep3}" / "panda_plus_leak_split.json"
    clean = EVAL_ROOT / f"ep{ep3}" / "panda_plus_clean30.json"
    isup = EVAL_ROOT / f"ep{ep3}" / "panda_plus_isup.csv"
    return leak.is_file() and clean.is_file() and isup.is_file()


def official_done(ckpt: Path) -> bool:
    out = SICAP_EVAL_ROOT / f"official_{ckpt.stem}" / "sicapv2_summary.json"
    return out.is_file() and out.stat().st_size > 0


def last_epoch_ckpt() -> Path | None:
    """Final epoch E: either FINAL_EPOCH env, or the max epoch on disk once
    TRAINING_COMPLETE.txt exists (full_train trains a fixed epoch count, no
    early stop, so the max epoch on disk IS epoch E)."""
    done = CKPT_DIR / "TRAINING_COMPLETE.txt"
    if not done.is_file():
        return None
    eps = list_epochs()
    if not eps:
        return None
    if FINAL_EPOCH:
        matches = [c for e, c in eps if e == FINAL_EPOCH]
        return matches[0] if matches else None
    return max(eps, key=lambda x: x[0])[1]


def sbatch(export: str, script: Path, *extra: str):
    cmd = ["sbatch", f"--export={export}", f"--exclude={EXCLUDE}", str(script), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    log(f"sbatch rc={r.returncode} out={out.strip()[:300]}")
    m = re.search(r"Submitted batch job (\d+)", out)
    return int(m.group(1)) if m else None


def main() -> None:
    log(f"START watch Stage-B tag={TAG} ckpt_dir={CKPT_DIR} (PANDA+=drift only; official=LAST epoch once)")
    state = load_state()
    while True:
        if not job_active(state.get("active_pp")):
            state["active_pp"] = None
        if not job_active(state.get("active_official")):
            state["active_official"] = None

        submitted_any = False
        for ep, ckpt in list_epochs():
            if state.get("active_pp") is None and ep not in state.get("submitted_pp", []) and not pp_done(ep):
                export = f"ALL,RUN_TAG={TAG},EPOCH={ep},LEAK_PREFIX={LEAK_PREFIX},EVAL_BS={EVAL_BS}"
                jid = sbatch(export, PP_SCRIPT, str(ckpt))
                if jid:
                    state.setdefault("submitted_pp", []).append(ep)
                    state["active_pp"] = jid
                    log(f"submitted PANDA+ ep{ep:03d} job={jid} (drift only)")
                    save_state(state)
                    submitted_any = True
                    break

        if not submitted_any and state.get("active_official") is None and not state.get("submitted_official"):
            last = last_epoch_ckpt()
            if last is not None and not official_done(last):
                out_dir = SICAP_EVAL_ROOT / f"official_{last.stem}"
                export = f"ALL,CKPT={last},PARTITION=official,EVAL_BS=8,OUT_DIR={out_dir}"
                jid = sbatch(export, SICAP_SCRIPT)
                if jid:
                    state["submitted_official"] = True
                    state["active_official"] = jid
                    state["official_ckpt"] = str(last)
                    log(f"submitted OFFICIAL Test once ckpt={last.name} job={jid} (= epoch E, last epoch)")
                    save_state(state)

        save_state(state)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
