"""Model 2 pseudo-label rule classification (A/B/C/D) from the ISUP diagnostic.

The script that originally produced ``correction_manifest.csv`` /
``rule_d_manifest.csv`` was lost from disk and was never committed to git (see
2026-07-27 session notes) -- only its *outputs* survived. This module rebuilds
the classification logic from first principles, matching the rule semantics
recorded in those surviving files, and implements the *new* Rule B design
(asymmetric loss weighting, not per-pixel soft targets -- see
``rule_b_asymmetric_weight_map`` in train/losses.py).

Rule definitions (mismatched slides only; matched slides need no correction):
    A: metadata_isup == 0                          -- tiny FP speckles, no fix
    B: (metadata_isup, derived_isup) in {(2,3),(3,2)} -- validated 2<->3 swap
    C_pure: model derived class(es) the metadata pattern does not contain
            ("invented" a class), AND metadata is a pure single pattern
            (e.g. "4+4") -- unambiguous, one possible correct answer
    C_two_class_unresolved: same "invented class" signature, but metadata has
            TWO expected classes (e.g. "4+5") -- which of the two absorbs the
            invented pixels is NOT validated. Left unchanged, same as D, until
            revisited (see PART 5 of the Model 2 loss design doc).
    D: model's derived pattern is missing an expected class entirely, with no
       invented class -- no reliable correction signal, mask left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from patch_utils import PROJECT

DIAGNOSTIC_REPORT_5PCT = PROJECT / "outputs" / "pseudo_label" / "diagnostic_report.csv"
DEFAULT_MANIFEST_OUT = PROJECT / "outputs" / "pseudo_label" / "model2_rule_manifest.csv"
DEFAULT_SUMMARY_OUT = PROJECT / "outputs" / "pseudo_label" / "model2_rule_manifest_summary.json"

RULE_B_ISUP_PAIRS = {(2, 3), (3, 2)}
CANCER_CLASSES = (3, 4, 5)


def gleason_to_classes(gleason: str) -> set[int]:
    """'4+3' -> {4, 3}. Empty set for negative/benign/unparseable strings."""
    g = str(gleason).strip().lower()
    if g in {"negative", "benign", "0+0", "nan", ""}:
        return set()
    if "+" not in g:
        raise ValueError(f"Unexpected gleason string: {gleason!r}")
    a, b = g.split("+", 1)
    return {int(a), int(b)}


def primary_gleason_class(gleason: str) -> int:
    return int(str(gleason).strip().split("+", 1)[0])


def classify_rule(
    *,
    metadata_gleason: str,
    metadata_isup: int,
    derived_gleason: str,
    derived_isup: int,
    match: bool,
) -> str:
    """Classify one mismatched slide into A/B/C_pure/C_two_class_unresolved/D.

    Returns "match" for slides that already agree (no correction needed).
    This function is exhaustive over all mismatched slides by construction --
    every branch returns, so counts always sum to total_slides.
    """
    if match:
        return "match"
    if int(metadata_isup) == 0:
        return "A"
    if (int(metadata_isup), int(derived_isup)) in RULE_B_ISUP_PAIRS:
        return "B"

    expected = gleason_to_classes(metadata_gleason)
    derived_classes = gleason_to_classes(derived_gleason)
    invented = derived_classes - expected
    missing = expected - derived_classes

    if invented:
        # Model predicted a cancer class with NO legitimate presence in the
        # metadata pattern at all. Unambiguous only when metadata has a single
        # expected class -- with two expected classes we don't know which one
        # the invented pixels should have been instead (see PART 5).
        return "C_pure" if len(expected) == 1 else "C_two_class_unresolved"
    if missing:
        # Model's derived pattern simply lacks a class the metadata expects,
        # without inventing anything new -- no reliable alternative target.
        return "D"
    # Same class set, different ISUP somehow (e.g. primary/secondary order
    # ambiguity outside the validated 2<->3 pair) -- treat conservatively as
    # no-action, same as D.
    return "D"


def build_manifest(
    diagnostic_csv: Path = DIAGNOSTIC_REPORT_5PCT,
    out_csv: Path = DEFAULT_MANIFEST_OUT,
    out_summary: Path = DEFAULT_SUMMARY_OUT,
) -> pd.DataFrame:
    df = pd.read_csv(diagnostic_csv)
    df["match"] = df["match"].astype(str).str.lower().isin(["true", "1"])

    df["rule"] = [
        classify_rule(
            metadata_gleason=r.metadata_gleason,
            metadata_isup=r.metadata_isup,
            derived_gleason=r.derived_gleason,
            derived_isup=r.derived_isup,
            match=r.match,
        )
        for r in df.itertuples()
    ]

    out = df[
        [
            "slide_id",
            "metadata_gleason",
            "metadata_isup",
            "derived_gleason",
            "derived_isup",
            "match",
            "rule",
        ]
    ].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    counts = out["rule"].value_counts().to_dict()
    summary = {
        "source": str(diagnostic_csv),
        "min_area_pct": 0.05,
        "total_slides": int(len(out)),
        "counts": {k: int(v) for k, v in counts.items()},
        "note": (
            "Rebuilt from diagnostic_report.csv after the original phase3 "
            "classification script was lost (never committed). "
            "C_two_class_unresolved slides get NO correction (same as D) "
            "per the Model 2 loss design doc PART 5 -- not yet validated."
        ),
    }
    import json

    out_summary.write_text(json.dumps(summary, indent=2))
    return out


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"Total slides: {total}")
    for rule, count in df["rule"].value_counts().items():
        print(f"  {rule:28s} {count:5d}  ({100.0 * count / total:.1f}%)")


if __name__ == "__main__":
    manifest = build_manifest()
    print_summary(manifest)
    print(f"\nWrote {DEFAULT_MANIFEST_OUT}")
    print(f"Wrote {DEFAULT_SUMMARY_OUT}")
