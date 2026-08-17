#!/usr/bin/env python3
"""Step 2: write PANDA / PANDA+ / ISUP / G5 into the registry and gate Step 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correction_pipeline import (  # noqa: E402
    FLAG_DICE_UP_ISUP_NOT,
    FLAG_MISSING_ISUP,
    REGISTRY_PATH,
    STATUS_NOT_VALIDATED,
    STATUS_VALIDATION_ONLY,
    assess_validation,
    collect_metrics,
    load_registry,
    upsert_model,
    write_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a registered teacher before referee")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    ap.add_argument(
        "--allow-step3-anyway",
        action="store_true",
        help="Do not use this to auto-train. Only overrides the Step-3 gate after manual review.",
    )
    args = ap.parse_args()

    payload = load_registry(args.registry)
    entry = payload.get("models", {}).get(args.model_id)
    if not entry:
        raise SystemExit(f"{args.model_id} is not in {args.registry}. Run register_correction_model.py first.")

    metrics = collect_metrics(entry)
    status, flags = assess_validation(
        metrics,
        payload.get("incumbent_teacher"),
        validation_only=entry.get("validation_status") == STATUS_VALIDATION_ONLY,
    )
    if args.allow_step3_anyway and status == STATUS_NOT_VALIDATED:
        flags = list(flags) + ["MANUAL_OVERRIDE_STEP3"]
        # Keep NOT_VALIDATED so comparison still shows the failure signature.
    saved = upsert_model(
        {
            "model_id": args.model_id,
            "metrics": metrics,
            "validation_status": status,
            "validation_flags": flags,
        },
        path=args.registry,
    )
    write_comparison(load_registry(args.registry))

    m = saved["metrics"]
    print(f"VALIDATE model_id={args.model_id} status={status} flags={flags or '[]'}")
    print(f"PANDA_val_dice={m.get('panda_val_cancer_dice')}")
    print(f"PANDA+_dice={m.get('panda_plus_cancer_dice')}")
    print(f"PANDA+_isup={m.get('panda_plus_isup_match')}")
    print(f"G5_precision={m.get('panda_plus_g5_precision')} G5_recall={m.get('panda_plus_g5_recall')}")
    if FLAG_DICE_UP_ISUP_NOT in flags:
        print(
            "BLOCK_STEP3 known failure signature: PANDA+ Dice improved over incumbent "
            "but PANDA+ ISUP did not. Manual review required before referee."
        )
    if FLAG_MISSING_ISUP in flags:
        print("BLOCK_STEP3 PANDA+ gland ISUP was not scored. Do not skip it because Dice looks good.")
    if status == STATUS_NOT_VALIDATED and not args.allow_step3_anyway:
        print(f"STEP3_HELD {args.model_id} flags={flags}")
        return
    print("STEP3_OK")


if __name__ == "__main__":
    main()
