#!/usr/bin/env python3
"""Section 3: nearest-allowed destinations, ties, examples. No training.

Reads the already-written correction_manifest.csv — does not touch H5s.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path("/common/omarmlab/members/anh/panda_project")
REF = PROJECT / "outputs" / "pseudo_label" / "corrections_opt3_omar6_locked_locked_r2_ep014"


def parse_allowed(raw) -> set[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    return {int(x) for x in text.split("|") if x != ""}


def parse_gleason(raw) -> tuple[int | None, int | None]:
    text = str(raw or "").strip()
    if "+" not in text:
        return None, None
    a, b = text.split("+", 1)
    try:
        return int(a), int(b)
    except ValueError:
        return None, None


def nearest_allowed(pred: int, allowed: set[int]) -> int:
    return min(allowed, key=lambda a: (abs(a - pred), a))


def main() -> None:
    man = pd.read_csv(REF / "correction_manifest.csv", dtype={"slide_id": str})
    done = man[man["skipped"].fillna("") == ""].copy()
    done["n_swap"] = done["n_swap"].fillna(0)
    swap = done[done["n_swap"] > 0].copy()

    totals = {
        "from_g3": int(done["n_swap_from_g3"].fillna(0).sum()),
        "from_g4": int(done["n_swap_from_g4"].fillna(0).sum()),
        "from_g5": int(done["n_swap_from_g5"].fillna(0).sum()),
        "to_g3": int(done["n_swap_to_g3"].fillna(0).sum()),
        "to_g4": int(done["n_swap_to_g4"].fillna(0).sum()),
        "to_g5": int(done["n_swap_to_g5"].fillna(0).sum()),
        "n_swap": int(done["n_swap"].fillna(0).sum()),
    }

    rows = []
    n_tie_g4_on_3and5 = 0
    n_slides_3and5 = 0
    n_slides_3and5_with_g4_swap = 0
    dest_primary = 0
    dest_secondary = 0
    dest_same = 0
    examples = []

    for rec in swap.itertuples():
        allowed = parse_allowed(getattr(rec, "allowed", ""))
        primary, secondary = parse_gleason(getattr(rec, "metadata_gleason", ""))
        froms = {
            3: int(getattr(rec, "n_swap_from_g3") or 0),
            4: int(getattr(rec, "n_swap_from_g4") or 0),
            5: int(getattr(rec, "n_swap_from_g5") or 0),
        }
        tos = {
            3: int(getattr(rec, "n_swap_to_g3") or 0),
            4: int(getattr(rec, "n_swap_to_g4") or 0),
            5: int(getattr(rec, "n_swap_to_g5") or 0),
        }
        if allowed == {3, 5}:
            n_slides_3and5 += 1
            if froms[4]:
                n_slides_3and5_with_g4_swap += 1
                n_tie_g4_on_3and5 += froms[4]
        for pred, n in froms.items():
            if n == 0 or pred in allowed or not allowed:
                continue
            dest = nearest_allowed(pred, allowed)
            if primary is not None and dest == primary == secondary:
                dest_same += n
            elif primary is not None and dest == primary:
                dest_primary += n
            elif secondary is not None and dest == secondary:
                dest_secondary += n
        rows.append(
            {
                "allowed": "|".join(str(c) for c in sorted(allowed)),
                "slides": 1,
                "n_swap": int(rec.n_swap),
                "from_g3": froms[3],
                "from_g4": froms[4],
                "from_g5": froms[5],
                "to_g3": tos[3],
                "to_g4": tos[4],
                "to_g5": tos[5],
            }
        )

    by_allowed = pd.DataFrame(rows).groupby("allowed", as_index=False).sum()
    by_allowed = by_allowed.sort_values("n_swap", ascending=False)

    # Examples: largest swap; a 3+5 with G4 ties; a G5-heavy; a G3-heavy
    swap_sorted = swap.sort_values("n_swap", ascending=False)
    pick = []
    if len(swap_sorted):
        r = swap_sorted.iloc[0]
        pick.append(("largest_swap", r))
    t35 = swap[swap["allowed"].astype(str) == "3|5"].sort_values("n_swap_from_g4", ascending=False)
    if len(t35):
        pick.append(("isup_3plus5_g4_tie", t35.iloc[0]))
    g5 = swap.sort_values("n_swap_from_g5", ascending=False)
    if len(g5):
        pick.append(("most_from_g5", g5.iloc[0]))
    g3 = swap.sort_values("n_swap_from_g3", ascending=False)
    if len(g3):
        pick.append(("most_from_g3", g3.iloc[0]))
    for kind, r in pick:
        allowed = parse_allowed(r.get("allowed"))
        examples.append(
            {
                "kind": kind,
                "slide_id": str(r["slide_id"]),
                "gleason": str(r.get("metadata_gleason", "")),
                "isup": int(r["metadata_isup"]) if pd.notna(r.get("metadata_isup")) else None,
                "allowed": "|".join(str(c) for c in sorted(allowed)),
                "n_swap": int(r["n_swap"]),
                "from_g3": int(r["n_swap_from_g3"] or 0),
                "from_g4": int(r["n_swap_from_g4"] or 0),
                "from_g5": int(r["n_swap_from_g5"] or 0),
                "to_g3": int(r["n_swap_to_g3"] or 0),
                "to_g4": int(r["n_swap_to_g4"] or 0),
                "to_g5": int(r["n_swap_to_g5"] or 0),
            }
        )

    n_swap = max(totals["n_swap"], 1)
    payload = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "nearest_allowed = min(allowed, key=lambda a: (abs(a-pred), a)) — lower grade on distance tie",
        "totals": totals,
        "share_from": {f"g{c}": totals[f"from_g{c}"] / n_swap for c in (3, 4, 5)},
        "share_to": {f"g{c}": totals[f"to_g{c}"] / n_swap for c in (3, 4, 5)},
        "destination_role": {
            "to_primary": dest_primary,
            "to_secondary": dest_secondary,
            "to_primary_equals_secondary": dest_same,
            "frac_to_primary": dest_primary / n_swap,
            "frac_to_secondary": dest_secondary / n_swap,
            "frac_to_same_both": dest_same / n_swap,
        },
        "ties_g4_on_3plus5": {
            "note": "allowed {3,5}, pred G4: distances tie, lower grade wins → G3",
            "n_slides_allowed_3_and_5_with_swap": n_slides_3and5,
            "n_slides_with_g4_swap": n_slides_3and5_with_g4_swap,
            "n_pixels_g4_to_g3_tie": n_tie_g4_on_3and5,
            "frac_of_all_swaps": n_tie_g4_on_3and5 / n_swap,
        },
        "by_allowed": by_allowed.to_dict(orient="records"),
        "examples": examples,
        "auto_train": False,
    }

    REF.joinpath("section3_nearest_allowed.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    by_allowed.to_csv(REF / "section3_by_allowed.csv", index=False)

    def pct(n: int) -> str:
        return f"{100.0 * n / n_swap:.2f}%"

    lines = [
        "# Section 3 — nearest-allowed destinations and ties",
        "",
        "Source: `correction_manifest.csv` from job 5513153. No H5 re-read. No training.",
        "",
        f"Tie-break: `min(allowed, key=lambda a: (abs(a-pred), a))` — **lower grade wins a distance tie**.",
        "",
        "## Where swaps come from / go",
        "",
        "| | pixels | share of 3.07B swaps |",
        "|---|---:|---:|",
        f"| from G3 | {totals['from_g3']:,} | {pct(totals['from_g3'])} |",
        f"| from G4 | {totals['from_g4']:,} | {pct(totals['from_g4'])} |",
        f"| from G5 | {totals['from_g5']:,} | {pct(totals['from_g5'])} |",
        f"| to G3 | {totals['to_g3']:,} | {pct(totals['to_g3'])} |",
        f"| to G4 | {totals['to_g4']:,} | {pct(totals['to_g4'])} |",
        f"| to G5 | {totals['to_g5']:,} | {pct(totals['to_g5'])} |",
        "",
        "## Primary vs secondary destination",
        "",
        f"- To clinical **primary**: {dest_primary:,} ({pct(dest_primary)})",
        f"- To clinical **secondary**: {dest_secondary:,} ({pct(dest_secondary)})",
        f"- To the only allowed grade (P=S, e.g. 4+4): {dest_same:,} ({pct(dest_same)})",
        "",
        "## The 3+5 / G4 tie",
        "",
        "On 3+5 slides allowed={3,5}. Pred G4 is equidistant; rule picks **G3**.",
        f"- Swap slides with allowed 3|5: **{n_slides_3and5}**",
        f"- Of those, slides with any G4→G3 tie swap: **{n_slides_3and5_with_g4_swap}**",
        f"- Tie pixels: **{n_tie_g4_on_3and5:,}** ({pct(n_tie_g4_on_3and5)} of all swaps)",
        "",
        "## By allowed set",
        "",
        "| allowed | slides | n_swap | from G3 | from G4 | from G5 | to G3 | to G4 | to G5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_allowed.itertuples():
        lines.append(
            f"| {r.allowed} | {int(r.slides):,} | {int(r.n_swap):,} | "
            f"{int(r.from_g3):,} | {int(r.from_g4):,} | {int(r.from_g5):,} | "
            f"{int(r.to_g3):,} | {int(r.to_g4):,} | {int(r.to_g5):,} |"
        )
    lines += ["", "## Examples", ""]
    for ex in examples:
        lines.append(
            f"- **{ex['kind']}** `{ex['slide_id']}` gleason {ex['gleason']} "
            f"allowed {ex['allowed']} n_swap={ex['n_swap']:,} "
            f"from G3/G4/G5={ex['from_g3']:,}/{ex['from_g4']:,}/{ex['from_g5']:,} "
            f"to G3/G4/G5={ex['to_g3']:,}/{ex['to_g4']:,}/{ex['to_g5']:,}"
        )
    lines += ["", "No training started.", ""]
    text = "\n".join(lines)
    (REF / "section3_nearest_allowed.md").write_text(text, encoding="utf-8")
    docs = PROJECT / "outputs" / "docs" / "opt3_this_run" / "section3_nearest_allowed.md"
    docs.write_text(text, encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "by_allowed"}, indent=2))
    print("Wrote", REF / "section3_nearest_allowed.md")


if __name__ == "__main__":
    main()
