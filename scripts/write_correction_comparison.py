#!/usr/bin/env python3
"""Step 4+5: copy G5-bias into the registry and rewrite the comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correction_pipeline import (  # noqa: E402
    FLAG_G5_BIAS,
    REGISTRY_PATH,
    g5_bias_from_report,
    load_registry,
    upsert_model,
    write_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Record G5-swap bias and refresh comparison")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    ap.add_argument("--balance-report", type=Path, default=None)
    args = ap.parse_args()

    payload = load_registry(args.registry)
    model_ids = [args.model_id] if args.model_id else list(payload.get("models", {}))
    for mid in model_ids:
        entry = payload["models"].get(mid)
        if not entry:
            raise SystemExit(f"missing {mid}")
        report_path = args.balance_report
        if report_path is None:
            report_path = Path(entry["paths"]["corrections"]) / "balance_report.json"
        if not report_path.is_file():
            print(f"SKIP {mid} no {report_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        g5 = g5_bias_from_report(report)
        flags = list(entry.get("validation_flags") or [])
        if g5.get("g5_bias_flag") and FLAG_G5_BIAS not in flags:
            flags.append(FLAG_G5_BIAS)
        upsert_model(
            {
                "model_id": mid,
                "g5_bias": g5,
                "validation_flags": flags,
            },
            path=args.registry,
        )
        print(
            f"G5_BIAS model_id={mid} swap_to_g5={g5.get('pct_high_conf_swaps_to_g5')}% "
            f"mask_g5={g5.get('pct_original_mask_g5')}% flagged={g5.get('g5_bias_flag')}"
        )
    write_comparison(load_registry(args.registry))
    print(f"comparison csv/md refreshed from {args.registry}")


if __name__ == "__main__":
    main()
