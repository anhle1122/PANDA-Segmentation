#!/usr/bin/env python3
"""Watch SICAP FT ckpt dir; submit PANDA+ leak-split + SICAP MPP×2 per new epoch."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
TAG = os.environ.get("RUN_TAG", "opt3_sicap_ft_r3ep6_val1_liveall").strip()
LEAK_PREFIX = os.environ.get("LEAK_PREFIX", "sicap_ft").strip()
FOLD = os.environ.get("SICAP_FOLD", "1").strip()
CKPT_DIR = PROJECT / "outputs" / "checkpoints" / f"uni2_upernet_raw_{TAG}"
EVAL_ROOT = PROJECT / "outputs" / "pseudo_label" / "epoch_eval" / TAG
LEAK_ROOT = (
    PROJECT
    / "outputs"
    / "docs"
    / "slide_groups"
    / "panda_plus_lookalike_gallery"
    / "leak_split"
)
SICAP_EVAL_ROOT = PROJECT / "outputs" / "evaluation" / "sicapv2"
STATE_PATH = PROJECT / "outputs" / "pseudo_label" / f"epoch_eval_watcher_{LEAK_PREFIX}_state.json"
LOG_PATH = PROJECT / "outputs" / "pseudo_label" / f"epoch_eval_watcher_{LEAK_PREFIX}.log"
PP_SCRIPT = PROJECT / "scripts" / "slurm_eval_opt3_panda_plus_r2redo.sh"
SICAP_SCRIPT = PROJECT / "scripts" / "slurm_eval_sicapv2_mpp.sh"
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
    return {"submitted_pp": [], "submitted_sicap": [], "active_pp": None, "active_sicap": None}


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


def job_active(job_id: int | None) -> bool:
    if not job_id:
        return False
    r = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = (r.stdout or "").strip().splitlines()
    return bool(lines) and lines[0] in {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "REQUEUED",
    }


def pp_done(ep: int) -> bool:
    ep3 = f"{ep:03d}"
    leak = LEAK_ROOT / f"{LEAK_PREFIX}_ep{ep3}" / "panda_plus_leak_split.json"
    clean = EVAL_ROOT / f"ep{ep3}" / "panda_plus_clean30.json"
    isup = EVAL_ROOT / f"ep{ep3}" / "panda_plus_isup.csv"
    return leak.is_file() and clean.is_file() and isup.is_file()


def sicap_done(ckpt: Path) -> bool:
    out = SICAP_EVAL_ROOT / f"mpp2x_{ckpt.stem}" / "sicapv2_summary.json"
    return out.is_file() and out.stat().st_size > 0


def sbatch(export: str, script: Path, *extra: str) -> int | None:
    cmd = [
        "sbatch",
        f"--export={export}",
        f"--exclude={EXCLUDE}",
        str(script),
        *extra,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    log(f"sbatch rc={r.returncode} out={out.strip()[:300]}")
    m = re.search(r"Submitted batch job (\d+)", out)
    return int(m.group(1)) if m else None


def main() -> None:
    log(f"START watch SICAP-FT tag={TAG} fold={FOLD} ckpt_dir={CKPT_DIR}")
    state = load_state()
    while True:
        if job_active(state.get("active_pp")):
            pass
        else:
            state["active_pp"] = None
        if job_active(state.get("active_sicap")):
            pass
        else:
            state["active_sicap"] = None

        for ep, ckpt in list_epochs():
            # PANDA+
            if (
                state.get("active_pp") is None
                and ep not in state.get("submitted_pp", [])
                and not pp_done(ep)
            ):
                export = (
                    f"ALL,RUN_TAG={TAG},EPOCH={ep},LEAK_PREFIX={LEAK_PREFIX},EVAL_BS={EVAL_BS}"
                )
                jid = sbatch(
                    export,
                    PP_SCRIPT,
                    str(ckpt),
                )
                if jid:
                    state.setdefault("submitted_pp", []).append(ep)
                    state["active_pp"] = jid
                    log(f"submitted PANDA+ ep{ep:03d} job={jid}")
                    save_state(state)
                    break

            # SICAP holdout fold (fair for Val{fold} FT)
            if (
                state.get("active_sicap") is None
                and ep not in state.get("submitted_sicap", [])
                and not sicap_done(ckpt)
            ):
                export = (
                    f"ALL,CKPT={ckpt},SCALE_FACTOR=2,FOLDS={FOLD},EVAL_BS=8"
                )
                jid = sbatch(export, SICAP_SCRIPT)
                if jid:
                    state.setdefault("submitted_sicap", []).append(ep)
                    state["active_sicap"] = jid
                    log(f"submitted SICAP MPP×2 ep{ep:03d} fold={FOLD} job={jid}")
                    save_state(state)
                    break

        save_state(state)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
