#!/usr/bin/env python3
"""Cheap sanity check: ep14 ISUP-referee vs original wmfix (teacher A Rules 1-3).

Does not train. Writes wmfix_vs_referee_summary.json/.md next to the referee out dir.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
WMFIX = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"
CORRECTING = {"rule1_soft_tie", "rule2_adjacent_invented", "rule3_invented_default"}
NO_REWRITE = {"match", "none", "wide_margin_unresolved", "rule1_wide_margin"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--referee-dir", type=Path, required=True)
    ap.add_argument("--wmfix", type=Path, default=WMFIX)
    args = ap.parse_args()

    ref_path = args.referee_dir / "correction_manifest.csv"
    if not ref_path.is_file():
        raise SystemExit("missing {0}".format(ref_path))
    balance_path = args.referee_dir / "balance_report.json"
    balance = {}
    if balance_path.is_file():
        balance = json.loads(balance_path.read_text(encoding="utf-8"))
    if not args.wmfix.is_file():
        raise SystemExit("missing wmfix manifest {0}".format(args.wmfix))

    ref = pd.read_csv(ref_path, dtype={"slide_id": str})
    wm = pd.read_csv(args.wmfix, dtype={"slide_id": str})

    ref["rewritten"] = (ref.get("n_swap", 0).fillna(0) > 0) | (
        ref.get("n_ignore", 0).fillna(0) > 0
    )
    ref["swapped"] = ref.get("n_swap", 0).fillna(0) > 0
    ref["skipped"] = ref.get("skipped", "").fillna("").astype(str)
    wm["correcting"] = wm["rule_applied"].isin(CORRECTING)

    merged = ref.merge(
        wm[["slide_id", "rule_applied", "correcting"]],
        on="slide_id",
        how="outer",
        indicator=True,
    )

    both = merged[merged["_merge"] == "both"]
    n_both = int(len(both))
    n_ref_only = int((merged["_merge"] == "left_only").sum())
    n_wm_only = int((merged["_merge"] == "right_only").sum())

    both_sw = both[both["swapped"].fillna(False)]
    both_corr = both[both["correcting"].fillna(False)]
    overlap_swap = both[both["swapped"].fillna(False) & both["correcting"].fillna(False)]
    ref_swap_not_wm = both[both["swapped"].fillna(False) & ~both["correcting"].fillna(False)]
    wm_corr_not_ref = both[both["correcting"].fillna(False) & ~both["swapped"].fillna(False)]

    wm_on_ref_swaps = Counter(
        both_sw["rule_applied"].fillna("missing").astype(str).tolist()
    )
    ref_skip = Counter(ref["skipped"].replace("", "applied").tolist())

    totals = {
        "n_slides_referee": int(len(ref)),
        "n_slides_wmfix": int(len(wm)),
        "n_slides_in_both": n_both,
        "n_ref_only": n_ref_only,
        "n_wm_only": n_wm_only,
        "n_isup0_skipped": int((ref["skipped"] == "isup0").sum()),
        "n_referee_swapped": int(ref["swapped"].fillna(False).sum()),
        "n_referee_any_rewrite_or_ignore": int(ref["rewritten"].fillna(False).sum()),
        "n_wmfix_correcting": int(wm["correcting"].sum()),
        "n_overlap_swap_and_wmfix_correcting": int(len(overlap_swap)),
        "n_referee_swap_but_wmfix_no_rewrite": int(len(ref_swap_not_wm)),
        "n_wmfix_correcting_but_referee_no_swap": int(len(wm_corr_not_ref)),
        "wmfix_rule_on_referee_swapped_slides": dict(wm_on_ref_swaps),
        "referee_skip_counts": dict(ref_skip),
        "n_swap_pixels": int(ref.get("n_swap", pd.Series(dtype=float)).fillna(0).sum()),
        "n_ignore_pixels": int(ref.get("n_ignore", pd.Series(dtype=float)).fillna(0).sum()),
        "n_agree_pixels": int(ref.get("n_agree", pd.Series(dtype=float)).fillna(0).sum()),
        "n_disagree_pixels": int(ref.get("n_disagree", pd.Series(dtype=float)).fillna(0).sum()),
    }
    if n_both:
        totals["frac_referee_swaps_also_wmfix_correcting"] = (
            len(overlap_swap) / max(len(both_sw), 1)
        )
        totals["frac_wmfix_correcting_also_referee_swap"] = (
            len(overlap_swap) / max(len(both_corr), 1)
        )

    verdict = (
        "broadly consistent"
        if totals.get("frac_referee_swaps_also_wmfix_correcting", 0) >= 0.5
        else "diverges meaningfully"
    )
    totals["verdict"] = verdict
    totals["written_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    totals["referee_dir"] = str(args.referee_dir)
    totals["wmfix_manifest"] = str(args.wmfix)
    totals["balance_report"] = {
        k: balance.get(k)
        for k in (
            "n_pixels",
            "n_agree",
            "n_disagree",
            "n_ignore",
            "n_high_conf_disagree",
            "n_swap",
            "n_swap_from_g3",
            "n_swap_from_g4",
            "n_swap_from_g5",
            "n_swap_to_g3",
            "n_swap_to_g4",
            "n_swap_to_g5",
            "n_isup0_skipped",
            "g5_bias",
            "pct_high_conf_swaps_to_g5",
            "conf_threshold",
        )
    }

    out_json = args.referee_dir / "wmfix_vs_referee_summary.json"
    out_md = args.referee_dir / "wmfix_vs_referee_summary.md"
    out_json.write_text(json.dumps(totals, indent=2), encoding="utf-8")

    lines = [
        "# Locked ep14 referee vs original wmfix (teacher A Rules 1-3)",
        "",
        "Referee is the **current** three-way ISUP gate (agree / ignore / illegal-swap).",
        "wmfix is the **old** Rule 1-3 set. This is a sanity overlap check, not a score.",
        "",
        "| | n |",
        "|---|---:|",
        "| Referee slides | {0} |".format(totals["n_slides_referee"]),
        "| wmfix slides | {0} |".format(totals["n_slides_wmfix"]),
        "| In both | {0} |".format(n_both),
        "| Referee swapped (n_swap>0) | {0} |".format(totals["n_referee_swapped"]),
        "| wmfix correcting (R1 soft / R2 / R3) | {0} |".format(totals["n_wmfix_correcting"]),
        "| Overlap (swap AND wmfix correcting) | {0} |".format(totals["n_overlap_swap_and_wmfix_correcting"]),
        "| Referee swap, wmfix no rewrite | {0} |".format(totals["n_referee_swap_but_wmfix_no_rewrite"]),
        "| wmfix correcting, referee no swap | {0} |".format(totals["n_wmfix_correcting_but_referee_no_swap"]),
        "| ISUP-0 skipped by referee | {0} |".format(totals["n_isup0_skipped"]),
        "",
        "Swap pixels: **{0:,}**. Ignore pixels: **{1:,}**.".format(
            totals["n_swap_pixels"], totals["n_ignore_pixels"]
        ),
        "",
        "Referee pixel totals (from balance_report.json): agree={0:,}, disagree={1:,}, ignore={2:,}, high-conf disagree={3:,}.".format(
            int(balance.get("n_agree") or totals["n_agree_pixels"]),
            int(balance.get("n_disagree") or totals["n_disagree_pixels"]),
            int(balance.get("n_ignore") or totals["n_ignore_pixels"]),
            int(balance.get("n_high_conf_disagree") or 0),
        ),
        "Swaps from G3/G4/G5: {0:,} / {1:,} / {2:,}. Swaps to G3/G4/G5: {3:,} / {4:,} / {5:,}.".format(
            int(balance.get("n_swap_from_g3") or 0),
            int(balance.get("n_swap_from_g4") or 0),
            int(balance.get("n_swap_from_g5") or 0),
            int(balance.get("n_swap_to_g3") or 0),
            int(balance.get("n_swap_to_g4") or 0),
            int(balance.get("n_swap_to_g5") or 0),
        ),
        "",
        "**Verdict vs wmfix:** {0} (different rule families; overlap is a sanity check, not a score).".format(
            verdict
        ),
        "",
        "No training started.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    docs = PROJECT / "outputs" / "docs" / "opt3_this_run" / "ep14_referee_pretrain_summary.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print("Wrote", out_json)
    print("Wrote", out_md)


if __name__ == "__main__":
    main()
