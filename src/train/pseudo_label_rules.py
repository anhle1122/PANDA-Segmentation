"""Rule 1-3 pseudo-label classification for iterative self-training.

Consumes the slide-level ISUP diagnostic produced by src/isup_diagnostic.py
(model-derived Gleason/ISUP vs. clinical metadata) and decides, per slide,
whether a correction applies and what that correction is. This logic is
identical in every round -- only WHICH model's diagnostic it reads changes.

Rules are checked IN ORDER; first match wins:

  Rule 1  validated 2<->3 Gleason boundary swap. Two sub-cases split on the
          model's own G3/G4 area margin:
            rule1_soft_tie            margin < 0.15 -- genuine near-tie, keep
                                      the model's own renormalized split as a
                                      soft target
            wide_margin_unresolved    margin >= 0.15 -- known G4-over-call bias,
                                      but NO reliable pixel signal for which
                                      G3/G4 pixels are real vs over-extended.
                                      NO correction applied (same effect as
                                      missing-class / none); distinct name so
                                      these are auditable and we do not silently
                                      erase the true secondary class.
  Rule 2  rule2_adjacent_invented -- primary correct, invented secondary is
          grade-adjacent to the true secondary. Fixed 95/5 cushion.
  Rule 3  rule3_invented_default -- any other invented class. Hard target to
          the slide's true primary.
  none    everything else: no correction (ISUP-0 mismatches, missing-class
          mismatches). These are NOT separate rules, just the default.

The manifest this writes is slide-level only. The per-pixel decision needs the
SOURCE model's predicted class at each pixel, which is cached separately by
scripts/cache_source_predictions.py and combined with these slide-level
parameters at training time (see train/pseudo_label_dataset.py).

Target encoding: every rule reduces to a two-class distribution over the
flagged pixels -- ``target_main_class`` at ``target_main_weight`` plus an
optional ``target_cushion_class`` at ``target_cushion_weight``. Hard targets
are just cushion_class = -1 and main_weight = 1.0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from patch_utils import PROJECT

DEFAULT_DIAGNOSTIC = PROJECT / "outputs" / "pseudo_label" / "diagnostic_report.csv"
DEFAULT_MANIFEST_OUT = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"

# Fixed across all rounds -- do not tune (see design doc PART 7).
MARGIN_THRESHOLD = 0.15
RULE2_MAIN_WEIGHT = 0.95
RULE2_CUSHION_WEIGHT = 0.05

# Empirically validated at exactly this ISUP boundary (91/91 and 158/158 real
# cases confirmed as genuine primary/secondary order swaps). 4<->5 was checked
# and had ZERO genuine swap candidates -- do not extend this set.
RULE_1_ISUP_PAIRS = {(2, 3), (3, 2)}
CANCER_CLASSES = (3, 4, 5)

BENIGN_GLEASON_STRINGS = {"negative", "benign", "0+0", "nan", "none", ""}

NO_ACTION = "none"
MATCH = "match"
RULE1_SOFT_TIE = "rule1_soft_tie"
# Deprecated correcting rule name -- wide-margin cases now use
# WIDE_MARGIN_UNRESOLVED (no pixel rewrite). Kept only so old manifests that
# still say rule1_wide_margin are treated as non-correcting.
RULE1_WIDE_MARGIN = "rule1_wide_margin"
WIDE_MARGIN_UNRESOLVED = "wide_margin_unresolved"
RULE2_ADJACENT_INVENTED = "rule2_adjacent_invented"
RULE3_INVENTED_DEFAULT = "rule3_invented_default"

# Buckets that must NOT rewrite any pixels in the dataset / cache.
NO_CORRECTION_RULES = frozenset(
    {MATCH, NO_ACTION, WIDE_MARGIN_UNRESOLVED, RULE1_WIDE_MARGIN}
)


def parse_gleason(gleason: str) -> tuple[int, int]:
    """'4+3' -> (4, 3). Benign/negative/unparseable -> (0, 0)."""
    g = str(gleason).strip().lower()
    if g in BENIGN_GLEASON_STRINGS:
        return 0, 0
    if "+" not in g:
        raise ValueError(f"Unexpected gleason string: {gleason!r}")
    primary, secondary = g.split("+", 1)
    return int(primary), int(secondary)


def gleason_to_classes(gleason: str) -> set[int]:
    """'4+3' -> {3, 4}. Benign/negative -> empty set."""
    primary, secondary = parse_gleason(gleason)
    if primary == 0 and secondary == 0:
        return set()
    return {primary, secondary}


@dataclass
class RuleAssignment:
    """One slide's correction, or the absence of one.

    ``flag_pred_classes`` lists the SOURCE model's predicted classes whose
    pixels get corrected. A pixel is flagged iff the cached source prediction
    at that pixel is in this list. Empty means nothing is flagged.
    """

    rule_applied: str
    flag_pred_classes: tuple[int, ...] = ()
    target_main_class: int = -1
    target_main_weight: float = 0.0
    target_cushion_class: int = -1
    target_cushion_weight: float = 0.0
    reason: str = ""


def _no_action(reason: str) -> RuleAssignment:
    return RuleAssignment(rule_applied=NO_ACTION, reason=reason)


def _hard_target(rule: str, flag_classes: tuple[int, ...], target: int, reason: str) -> RuleAssignment:
    return RuleAssignment(
        rule_applied=rule,
        flag_pred_classes=flag_classes,
        target_main_class=target,
        target_main_weight=1.0,
        target_cushion_class=-1,
        target_cushion_weight=0.0,
        reason=reason,
    )


def classify_slide(
    *,
    metadata_gleason: str,
    metadata_isup: int,
    derived_gleason: str,
    derived_isup: int,
    match: bool,
    cancer_frac_g3: float,
    cancer_frac_g4: float,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> RuleAssignment:
    """Apply Rules 1-3 in order to one slide. First match wins."""
    if match:
        return RuleAssignment(rule_applied=MATCH, reason="derived ISUP already agrees with metadata")

    metadata_isup = int(metadata_isup)
    derived_isup = int(derived_isup)

    # ISUP-0 mismatches are formula-sensitivity artifacts on slides with no
    # legitimate cancer pattern to correct toward -- there is no true_primary
    # to use as a target. Guarding here keeps them out of Rule 3, which would
    # otherwise fire on every predicted cancer pixel (expected_classes is empty
    # for benign metadata) and "correct" them all to class 0.
    if metadata_isup == 0:
        return _no_action("metadata ISUP-0; tiny FP speckle artifact, no reliable target")

    true_primary, true_secondary = parse_gleason(metadata_gleason)
    expected = gleason_to_classes(metadata_gleason)
    if not expected:
        return _no_action("metadata Gleason has no cancer pattern to target")

    derived_classes = gleason_to_classes(derived_gleason)
    invented = sorted(derived_classes - expected)

    # ---- Rule 1: validated 2<->3 boundary swap ----
    if (metadata_isup, derived_isup) in RULE_1_ISUP_PAIRS:
        g3_pct = float(cancer_frac_g3)
        g4_pct = float(cancer_frac_g4)
        margin = abs(g3_pct - g4_pct)
        total = g3_pct + g4_pct
        if margin < margin_threshold and total > 0.0:
            # Genuine near-tie: the model's own split is self-consistent, so
            # keep it as a soft target rather than inventing a hard answer we
            # cannot verify per-pixel.
            return RuleAssignment(
                rule_applied=RULE1_SOFT_TIE,
                flag_pred_classes=(3, 4),
                target_main_class=3,
                target_main_weight=g3_pct / total,
                target_cushion_class=4,
                target_cushion_weight=g4_pct / total,
                reason=f"2<->3 swap, near-tie margin={margin:.3f}",
            )
        # Wide margin: we know the model over-calls one side of the 2<->3
        # boundary, but we have no reliable pixel-level signal for which G3/G4
        # pixels are the true secondary vs over-extension. Hard-correcting the
        # over-extended class erases the real secondary pattern. Leave the
        # original mask untouched -- same effect as missing-class / none -- but
        # tag explicitly so this unresolved bucket stays auditable.
        derived_primary, _ = parse_gleason(derived_gleason)
        return RuleAssignment(
            rule_applied=WIDE_MARGIN_UNRESOLVED,
            reason=(
                f"2<->3 swap, wide margin={margin:.3f}; over-extended class="
                f"{derived_primary} but no reliable pixel split -- NO ACTION "
                f"(do not erase true secondary)"
            ),
        )

    if not invented:
        # No invented class: the derived pattern is missing an expected class,
        # or reorders it outside the validated 2<->3 pair. Neither has a
        # reliable alternative target -- trusting the existing label is safer.
        return _no_action("no invented class (missing-class or unvalidated reorder)")

    # ---- Rule 2: invented class adjacent to the true secondary ----
    derived_primary, derived_secondary = parse_gleason(derived_gleason)
    primary_correct = derived_primary == true_primary
    invented_secondary = derived_secondary in invented
    is_adjacent = abs(derived_secondary - true_secondary) == 1
    # Pure patterns ("4+4") are deliberately excluded even when the invented
    # class is adjacent. Rule 2's reasoning is that the model confused a real
    # secondary pattern with its visual neighbour -- on a pure slide there is
    # no distinct secondary region to have confused, and the true answer is
    # unambiguous, so those belong in Rule 3 as a hard target rather than
    # getting a 5% cushion toward a class that provably is not there.
    has_distinct_secondary = true_secondary != true_primary
    if primary_correct and invented_secondary and is_adjacent and has_distinct_secondary:
        # G4/G5 (or any adjacent pair) look alike -- more plausible the model
        # confused the true secondary with its visual neighbour than that it
        # hallucinated an unrelated pattern. Lower confidence than Rule 1's
        # validated swap, hence a fixed cushion rather than a hard target, and
        # its own rule_applied value so these can be audited separately.
        # There is NO model percentage split to blend from here: the model
        # predicted ~0% of the true secondary, which is why it counts as
        # invented. The 95/5 is a fixed hedge, not model-derived.
        return RuleAssignment(
            rule_applied=RULE2_ADJACENT_INVENTED,
            flag_pred_classes=(derived_secondary,),
            target_main_class=true_secondary,
            target_main_weight=RULE2_MAIN_WEIGHT,
            target_cushion_class=derived_secondary,
            target_cushion_weight=RULE2_CUSHION_WEIGHT,
            reason=(
                f"primary {true_primary} correct; invented {derived_secondary} "
                f"adjacent to true secondary {true_secondary}"
            ),
        )

    # ---- Rule 3: any other invented class ----
    flag_classes = tuple(c for c in CANCER_CLASSES if c not in expected)
    if not flag_classes:
        return _no_action("invented set empty after restricting to cancer classes")
    confidence = "pure pattern, unambiguous" if len(expected) == 1 else "two expected classes, primary fallback"
    return _hard_target(
        RULE3_INVENTED_DEFAULT,
        flag_classes,
        true_primary,
        f"invented {invented} not in expected {sorted(expected)} ({confidence})",
    )


def build_manifest(
    diagnostic_csv: Path = DEFAULT_DIAGNOSTIC,
    out_csv: Path = DEFAULT_MANIFEST_OUT,
    *,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Classify every slide in the diagnostic and write the round's manifest."""
    df = pd.read_csv(diagnostic_csv, dtype={"slide_id": str})
    df["match"] = df["match"].astype(str).str.lower().isin(["true", "1"])

    rows = []
    for r in df.itertuples():
        assignment = classify_slide(
            metadata_gleason=r.metadata_gleason,
            metadata_isup=r.metadata_isup,
            derived_gleason=r.derived_gleason,
            derived_isup=r.derived_isup,
            match=r.match,
            cancer_frac_g3=r.cancer_frac_g3,
            cancer_frac_g4=r.cancer_frac_g4,
            margin_threshold=margin_threshold,
        )
        true_primary, true_secondary = parse_gleason(r.metadata_gleason)
        record = asdict(assignment)
        record["flag_pred_classes"] = "|".join(str(c) for c in assignment.flag_pred_classes)
        rows.append(
            {
                "slide_id": r.slide_id,
                **record,
                "metadata_gleason": r.metadata_gleason,
                "metadata_isup": int(r.metadata_isup),
                "derived_gleason": r.derived_gleason,
                "derived_isup": int(r.derived_isup),
                "match": bool(r.match),
                "true_primary": true_primary,
                "true_secondary": true_secondary,
                "cancer_frac_g3": float(r.cancer_frac_g3),
                "cancer_frac_g4": float(r.cancer_frac_g4),
                "cancer_frac_g5": float(r.cancer_frac_g5),
                "g3_g4_margin": abs(float(r.cancer_frac_g3) - float(r.cancer_frac_g4)),
            }
        )

    manifest = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)

    summary = {
        "source_diagnostic": str(diagnostic_csv),
        "margin_threshold": margin_threshold,
        "rule2_cushion": [RULE2_MAIN_WEIGHT, RULE2_CUSHION_WEIGHT],
        "total_slides": int(len(manifest)),
        "counts": {k: int(v) for k, v in manifest["rule_applied"].value_counts().items()},
        "correcting_rules": [
            RULE1_SOFT_TIE,
            RULE2_ADJACENT_INVENTED,
            RULE3_INVENTED_DEFAULT,
        ],
        "no_correction_rules": sorted(NO_CORRECTION_RULES),
    }
    summary_path = out_csv.with_name(out_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest


def print_summary(manifest: pd.DataFrame) -> None:
    total = len(manifest)
    print(f"Total slides: {total}")
    corrected = 0
    for rule, count in manifest["rule_applied"].value_counts().items():
        tag = "" if rule not in NO_CORRECTION_RULES else "  [no pixel rewrite]"
        print(f"  {rule:26s} {count:5d}  ({100.0 * count / total:.1f}%){tag}")
        if rule not in NO_CORRECTION_RULES:
            corrected += count
    print(f"\nSlides receiving a correction: {corrected} ({100.0 * corrected / total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Rule 1-3 pseudo-label manifest")
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--margin-threshold", type=float, default=MARGIN_THRESHOLD)
    args = parser.parse_args()

    manifest = build_manifest(
        args.diagnostic, args.out, margin_threshold=args.margin_threshold,
    )
    print_summary(manifest)
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.out.with_name(args.out.stem + '_summary.json')}")


if __name__ == "__main__":
    main()
