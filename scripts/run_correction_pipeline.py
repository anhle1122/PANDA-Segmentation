#!/usr/bin/env python3
"""Orchestrate Steps 0–5 for one checkpoint. Never submits training.

Cache jobs are queued one-at-a-time on L40S via --dependency=afterany.
Referee is CPU, afterok on the cache job, and only if validation passed
(or --allow-step3-anyway after manual review).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correction_pipeline import (  # noqa: E402
    KNOWN_JOBS,
    REGISTRY_PATH,
    STATUS_NOT_VALIDATED,
    STATUS_VALIDATION_ONLY,
    assess_validation,
    collect_metrics,
    identity_entry,
    load_registry,
    pack_exists_for_tag_epoch,
    parse_epoch_from_ckpt,
    resolve_checkpoint,
    teacher_pack_dir,
    upsert_model,
    write_comparison,
)

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
CACHE_SCRIPT = PROJECT / "scripts" / "slurm_cache_teacher_pack.sh"
AFTER_SCRIPT = PROJECT / "scripts" / "slurm_correction_after_cache.sh"


def _sbatch(cmd: list[str]) -> str:
    print("SBATCH", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        raise SystemExit(f"sbatch failed rc={r.returncode} stdout={out!r} stderr={(r.stderr or '').strip()}")
    job_id = out.split(";")[0].strip()
    if not job_id.isdigit():
        raise SystemExit(f"unparsable sbatch stdout={out!r}")
    return job_id


def latest_l40s_job() -> str | None:
    """Chain behind any current L40S cache/eval so we do not double-fire."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "lea14"), "-h", "-o", "%i %j %T %b"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    last = None
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        job_id, name, state, tres = parts[0], parts[1], parts[2], " ".join(parts[3:])
        if state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            continue
        l40 = "l40s" in tres.lower() or name in {"teacher_pack", "opt3_ep_eval", "opt3_eval"}
        if l40 or name in {"teacher_pack", "opt3_ep_eval"}:
            last = job_id
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Register + queue cache/referee for one checkpoint")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--source-job-id", required=True)
    ap.add_argument("--recipe-version", default="")
    ap.add_argument("--code-commit", default="")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--submit-jobs", action="store_true")
    ap.add_argument("--allow-step3-anyway", action="store_true")
    ap.add_argument("--after-job", default="", help="Optional extra afterany:JOBID for the L40S cache")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    known = KNOWN_JOBS.get(str(args.source_job_id), {})
    run_tag = args.run_tag or known.get("run_tag") or ""
    recipe = args.recipe_version or known.get("recipe_version") or ""
    commit = args.code_commit or known.get("code_commit") or ""
    note = known.get("code_commit_note", "")
    if not run_tag or not recipe or not commit:
        raise SystemExit("need run-tag / recipe-version / code-commit (or a known source-job-id)")

    if args.checkpoint is None and args.epoch is None:
        raise SystemExit("pass --checkpoint or --epoch")
    if args.checkpoint is not None:
        ckpt = resolve_checkpoint(args.checkpoint, run_tag, int(args.epoch or 0))
    else:
        ckpt = resolve_checkpoint(None, run_tag, int(args.epoch))
    epoch = args.epoch if args.epoch is not None else parse_epoch_from_ckpt(ckpt)
    entry = identity_entry(
        checkpoint=ckpt,
        run_tag=run_tag,
        recipe_version=recipe,
        source_job_id=str(args.source_job_id),
        code_commit=commit,
        code_commit_note=note,
        epoch=epoch,
        validation_only=args.validation_only,
    )
    metrics = collect_metrics(entry)
    status, flags = assess_validation(
        metrics,
        load_registry(args.registry).get("incumbent_teacher"),
        validation_only=args.validation_only,
    )
    entry["metrics"] = metrics
    entry["validation_status"] = status
    entry["validation_flags"] = flags
    saved = upsert_model(entry, path=args.registry)
    write_comparison(load_registry(args.registry))

    pack = teacher_pack_dir(run_tag, recipe, epoch)
    print(f"REGISTERED {saved['model_id']}")
    print(f"ckpt={saved['checkpoint_filename']} job={saved['source_job_id']} commit={saved['code_commit']}")
    print(f"recipe={saved['recipe_version']} pack={pack}")
    print(f"VALIDATE status={status} flags={flags or '[]'}")
    print(
        f"PANDA_val={metrics.get('panda_val_cancer_dice')} "
        f"PANDA+={metrics.get('panda_plus_cancer_dice')} "
        f"PANDA+_ISUP={metrics.get('panda_plus_isup_match')} "
        f"G5_P={metrics.get('panda_plus_g5_precision')} "
        f"G5_R={metrics.get('panda_plus_g5_recall')}"
    )
    if note:
        print(f"commit_note={note}")
    print("AUTO_TRAIN=false (Round N+1 is a manual pick from correction_comparison.md)")

    skip_cache = pack_exists_for_tag_epoch(pack)
    if skip_cache:
        print(f"SKIP_CACHE pack already exists: {pack}")

    if not args.submit_jobs:
        print("NO_SUBMIT (pass --submit-jobs to sbatch cache/referee)")
        return
    if args.dry_run:
        print("DRY_RUN no sbatch")
        return

    cache_job = None
    if not skip_cache:
        export = (
            f"ALL,RUN_TAG={run_tag},RECIPE_VERSION={recipe},"
            f"PACK_TAG={saved['pack_tag']},OUT_DIR={pack}"
        )
        cmd = [
            "sbatch",
            "--parsable",
            "--job-name=teacher_pack",
            "--gres=gpu:l40s:1",
            f"--export={export}",
        ]
        after = args.after_job or latest_l40s_job()
        if after:
            cmd.append(f"--dependency=afterany:{after}")
            print(f"CACHE_CHAIN afterany:{after}")
        cmd.extend([str(CACHE_SCRIPT), str(ckpt)])
        cache_job = _sbatch(cmd)
        upsert_model({"model_id": saved["model_id"], "jobs": {"cache": cache_job}}, path=args.registry)
        print(f"QUEUED_CACHE job={cache_job}")
    else:
        print("CACHE not queued")

    hold_step3 = status in {STATUS_NOT_VALIDATED, STATUS_VALIDATION_ONLY} and not args.allow_step3_anyway
    if status == STATUS_VALIDATION_ONLY:
        print("STEP3_HELD validation-only pack — referee needs --allow-validation-only after review")
        return
    if hold_step3:
        print(f"STEP3_HELD status={status} flags={flags} (no referee sbatch)")
        return

    export = (
        f"ALL,MODEL_ID={saved['model_id']},REGISTRY={args.registry},"
        f"ALLOW_STEP3_ANYWAY={'1' if args.allow_step3_anyway else '0'}"
    )
    cmd = ["sbatch", "--parsable", "--job-name=isup_referee", f"--export={export}"]
    if cache_job:
        cmd.append(f"--dependency=afterok:{cache_job}")
    elif skip_cache:
        pass
    cmd.append(str(AFTER_SCRIPT))
    ref_job = _sbatch(cmd)
    upsert_model({"model_id": saved["model_id"], "jobs": {"referee": ref_job}}, path=args.registry)
    print(f"QUEUED_REFEREE job={ref_job} (CPU; afterok cache if queued)")


if __name__ == "__main__":
    main()
