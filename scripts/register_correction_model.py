#!/usr/bin/env python3
"""Step 0: record checkpoint identity in model_registry.json before any cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correction_pipeline import (  # noqa: E402
    KNOWN_JOBS,
    REGISTRY_PATH,
    identity_entry,
    parse_epoch_from_ckpt,
    resolve_checkpoint,
    upsert_model,
    write_comparison,
    load_registry,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a checkpoint for between-round correction")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--source-job-id", required=True)
    ap.add_argument("--recipe-version", default="")
    ap.add_argument("--code-commit", default="")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = ap.parse_args()

    known = KNOWN_JOBS.get(str(args.source_job_id), {})
    run_tag = args.run_tag or known.get("run_tag") or ""
    recipe = args.recipe_version or known.get("recipe_version") or ""
    commit = args.code_commit or known.get("code_commit") or ""
    note = known.get("code_commit_note", "")
    if not run_tag or not recipe or not commit:
        raise SystemExit(
            "Need --run-tag, --recipe-version, and --code-commit "
            "(or a known --source-job-id in KNOWN_JOBS)."
        )
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
    saved = upsert_model(entry, path=args.registry)
    write_comparison(load_registry(args.registry))
    print(
        f"REGISTERED model_id={saved['model_id']} "
        f"ckpt={saved['checkpoint_filename']} job={saved['source_job_id']} "
        f"commit={saved['code_commit']} recipe={saved['recipe_version']}"
    )
    print(f"teacher_pack={saved['paths']['teacher_pack']}")
    print(f"corrections={saved['paths']['corrections']}")
    print(f"registry={args.registry}")
    if note:
        print(f"commit_note={note}")


if __name__ == "__main__":
    main()
