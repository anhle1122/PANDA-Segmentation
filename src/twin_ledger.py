#!/usr/bin/env python3
"""Canonical twin ledger: the single source of truth for slide dedupe.

Two frozen lists live under ``outputs/docs/slide_duplicates``:

* ``CANONICAL_twin_drops.csv``  -- slides removed as duplicates. Once here, a
  slide must never reappear in the splits, whatever a later bulk restore says.
* ``CANONICAL_not_twins.csv``   -- slides the user reviewed and kept. These are
  protected from any automatic drop.

Use ``--rebuild`` after a review session to fold new decisions in, and
``--verify`` (default) before training or before trusting a rescan gallery.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from patch_utils import PROJECT

DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
RESCAN = PROJECT / "outputs" / "docs" / "slide_duplicates_rescan_alive"
SPLITS = PROJECT / "outputs" / "splits"

LEDGER_DROPS = DUP / "CANONICAL_twin_drops.csv"
LEDGER_KEEPS = DUP / "CANONICAL_not_twins.csv"
LEDGER_PAIRS = DUP / "CANONICAL_not_twin_pairs.csv"

HEX32 = re.compile(r"^[0-9a-f]{32}$")

# Precedence tiers. A higher tier always overrides a lower one, so a review
# session can correct anything an earlier automatic pass decided.
TIER_AUTO = 1  # policy passes: safe/lower thresholds, greedy multi, galleries
TIER_REVIEW = 2  # user review of the first-round pair galleries
TIER_MULTI = 3  # user review of the multi-cluster pages (C1-35, C35-55)
TIER_LATEST = 4  # rescan-era decisions (2026-08-09 onwards)


def _ids(values) -> list[str]:
    return [str(v) for v in values if HEX32.match(str(v).lower())]


def collect_not_twins() -> pd.DataFrame:
    """Slides the user explicitly kept after looking at them."""
    rows = []

    def add(ids, source, tier=TIER_REVIEW):
        for i in _ids(ids):
            rows.append({"image_id": i, "source": source, "tier": tier})

    for name in (
        "user_marked_not_same_lower_iou.csv",
        "user_marked_not_same_safe_pairs.csv",
        "user_marked_not_same_lower_iou_forced_train.csv",
        "user_marked_not_same_soft24_regroup.csv",
    ):
        path = DUP / name
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            for col in df.columns:
                add(df[col].dropna(), name)

    for path in DUP.glob("user_*not_twin*.json"):
        add(re.findall(r"[0-9a-f]{32}", path.read_text().lower()), path.name)

    decisions = DUP / "user_multi_C1_35_final_decisions.csv"
    if decisions.exists():
        df = pd.read_csv(decisions, dtype=str)
        add(
            df.loc[df.action.str.contains("keep", case=False, na=False), "image_id"],
            decisions.name,
            TIER_MULTI,
        )

    for path in sorted(RESCAN.glob("**/*decisions*.csv")):
        df = pd.read_csv(path, dtype=str)
        if "action" not in df.columns:
            continue
        df = df[df.action.str.contains("keep", case=False, na=False)]
        for col in ("image_id", "keep_id", "drop_id"):
            if col in df.columns:
                add(df[col].dropna(), f"rescan:{path.name}", TIER_LATEST)

    return pd.DataFrame(rows)


def collect_not_twin_pairs() -> pd.DataFrame:
    """Specific pairs the user looked at and judged *not* twins of each other.

    Slide-level keeps are not enough to suppress a rescan edge: a slide that is
    not the twin of A can still be the twin of B (that is how 68a/8b126 was
    caught). So we track the adjudicated edge, not the slide.
    """
    rows = []

    def add(a, b, source):
        a, b = str(a).lower(), str(b).lower()
        if HEX32.match(a) and HEX32.match(b) and a != b:
            lo, hi = sorted([a, b])
            rows.append({"id_a": lo, "id_b": hi, "source": source})

    for name in (
        "user_marked_not_same_lower_iou.csv",
        "user_marked_not_same_safe_pairs.csv",
        "user_marked_not_same_lower_iou_forced_train.csv",
        "user_restore_multi_not_twins.csv",
        "user_marked_not_same_soft24_regroup.csv",
    ):
        path = DUP / name
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            for r in df.itertuples(index=False):
                add(r.keep_id, r.drop_id, name)

    for name in ("user_not_twins_p414_827_828_837_839.json", "user_safe_not_twins_p476_480_511.json"):
        path = DUP / name
        if path.exists():
            blob = json.loads(path.read_text())
            for entry in blob.get("restored", []) + blob.get("pairs", []):
                add(entry["keep_id"], entry["drop_id"], name)

    for path in sorted(RESCAN.rglob("*decisions*.csv")):
        df = pd.read_csv(path, dtype=str)
        if not {"keep_id", "drop_id"} <= set(df.columns) or "action" not in df.columns:
            continue
        df = df[df.action.str.contains("keep", case=False, na=False)]
        for r in df.itertuples(index=False):
            add(r.keep_id, r.drop_id, f"rescan:{path.name}")

    return pd.DataFrame(rows, columns=["id_a", "id_b", "source"])


def collect_drops() -> pd.DataFrame:
    """Slides removed as duplicates, from every decision log we have."""
    rows = []

    def add(ids, source, note="", tier=TIER_AUTO):
        for i in _ids(ids):
            rows.append({"image_id": i, "source": source, "note": note, "tier": tier})

    implied = DUP / "lower_iou_implied_twins_decisions.csv"
    if implied.exists():
        add(pd.read_csv(implied, dtype=str).drop_id, "lower_iou_implied_twin", "unmarked pair = twin")

    safe = DUP / "dedupe_safe_pairs_iou70.csv"
    not_same = DUP / "user_marked_not_same_safe_pairs.csv"
    if safe.exists():
        df = pd.read_csv(safe, dtype=str)
        skip = set()
        if not_same.exists():
            nt = pd.read_csv(not_same, dtype=str)
            skip = {tuple(sorted([r.keep_id, r.drop_id])) for r in nt.itertuples(index=False)}
        for r in df.itertuples(index=False):
            if tuple(sorted([r.keep_id, r.drop_id])) not in skip:
                add([r.drop_id], "safe_iou70_twin", f"keep {r.keep_id[:12]}")

    decisions = DUP / "user_multi_C1_35_final_decisions.csv"
    if decisions.exists():
        df = pd.read_csv(decisions, dtype=str)
        add(
            df.loc[df.action.str.contains("drop", case=False, na=False), "image_id"],
            "user_multi_drop",
            tier=TIER_MULTI,
        )

    legacy = DUP / "legacy_forced_twin_drops_decisions.csv"
    if legacy.exists():
        add(pd.read_csv(legacy, dtype=str).image_id, "legacy_forced_drop", tier=TIER_MULTI)

    friday = DUP / "user_restore_multi_safe_C35_55_friday_summary.json"
    if friday.exists():
        explicit = json.loads(friday.read_text()).get("still_dropped_explicit_user") or []
        if isinstance(explicit, dict):
            explicit = list(explicit)
        add(explicit, "user_multi_C35_55_explicit_drop", tier=TIER_MULTI)

    confirmed = DUP / "galleries" / "confirmed_twins_keep_drop" / "all_confirmed_twins_keep_drop.csv"
    if confirmed.exists():
        df = pd.read_csv(confirmed, dtype=str)
        for r in df.itertuples(index=False):
            add([r.drop_id], "confirmed_twins_gallery", f"{r.pair_id} keep {str(r.keep_id)[:12]}")

    for path in sorted(RESCAN.rglob("*decisions*.csv")) + [RESCAN / "reconcile_confirmed_twins" / "PROPOSAL_A_final_drops.csv"]:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        if "action" in df.columns:
            df = df[df.action.str.contains("drop", case=False, na=False)]
        if "drop_id" in df.columns:
            add(df.drop_id.dropna(), f"rescan:{path.name}", tier=TIER_LATEST)

    return pd.DataFrame(rows)


def _collapse(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("image_id")
        .agg(
            tier=("tier", "max"),
            sources=("source", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )


def rebuild() -> dict:
    keeps = _collapse(collect_not_twins())
    drops = _collapse(collect_drops())

    # Highest tier wins; on a tie a keep beats a drop, since keeping a possible
    # duplicate only costs data, while dropping a unique slide loses it.
    merged = keeps.merge(drops, on="image_id", how="outer", suffixes=("_keep", "_drop"))
    keep_tier = merged.tier_keep.fillna(0)
    drop_tier = merged.tier_drop.fillna(0)
    is_drop = drop_tier > keep_tier

    final_drops = merged[is_drop][["image_id", "sources_drop", "tier_drop"]].rename(
        columns={"sources_drop": "sources", "tier_drop": "tier"}
    )
    final_keeps = merged[~is_drop & (keep_tier > 0)][["image_id", "sources_keep", "tier_keep"]].rename(
        columns={"sources_keep": "sources", "tier_keep": "tier"}
    )
    for frame in (final_drops, final_keeps):
        frame["frozen_on"] = str(date.today())

    pairs = (
        collect_not_twin_pairs()
        .groupby(["id_a", "id_b"])
        .source.apply(lambda s: "|".join(sorted(set(s))))
        .reset_index()
    )
    pairs["frozen_on"] = str(date.today())

    final_drops.sort_values("image_id").to_csv(LEDGER_DROPS, index=False)
    final_keeps.sort_values("image_id").to_csv(LEDGER_KEEPS, index=False)
    pairs.to_csv(LEDGER_PAIRS, index=False)
    overridden = int((keep_tier.gt(0) & drop_tier.gt(0)).sum())
    return {
        "n_drops": len(final_drops),
        "n_not_twins": len(final_keeps),
        "n_adjudicated_not_twin_pairs": len(pairs),
        "conflicts_resolved_by_tier": overridden,
    }


def adjudicated_pairs() -> set[tuple[str, str]]:
    """Sorted (id_a, id_b) edges already judged not-twins; safe to hide."""
    if not LEDGER_PAIRS.exists():
        return set()
    df = pd.read_csv(LEDGER_PAIRS, dtype=str)
    return {(r.id_a, r.id_b) for r in df.itertuples(index=False)}


def verify() -> dict:
    if not LEDGER_DROPS.exists():
        raise SystemExit(f"No ledger yet: {LEDGER_DROPS} (run with --rebuild)")
    drops = pd.read_csv(LEDGER_DROPS, dtype=str)
    keeps = pd.read_csv(LEDGER_KEEPS, dtype=str) if LEDGER_KEEPS.exists() else pd.DataFrame(columns=["image_id"])
    splits = pd.read_csv(SPLITS / "panda_slide_splits.csv", dtype=str)
    alive = set(splits.image_id)

    resurrected = sorted(set(drops.image_id) & alive)
    missing_keeps = sorted(set(keeps.image_id) - alive)
    return {
        "alive_slides": len(alive),
        "ledger_drops": len(drops),
        "ledger_not_twins": len(keeps),
        "resurrected_twins_alive": resurrected,
        "not_twins_wrongly_dropped": missing_keeps,
        "clean": not resurrected and not missing_keeps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Fold current decision logs into the ledger")
    args = ap.parse_args()

    if args.rebuild:
        print(json.dumps(rebuild(), indent=2))

    report = verify()
    print(json.dumps(report, indent=2))
    if not report["clean"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
