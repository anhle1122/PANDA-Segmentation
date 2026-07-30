"""Round-to-round control logic for iterative pseudo-label self-training.

Pure decision functions, no I/O and no torch -- the loop driver calls these
between rounds, and the smoke test can exercise them with synthetic numbers.

Two independent checks run after every round from Round 2 on:

  bias_too_heavy()  -- is training against the previous model's raw predictions
                       as the *base* seg_target actively hurting external
                       (PANDA+) performance? If so the base seg_target reverts
                       to the ORIGINAL MASK permanently for the rest of the run.
  should_stop()     -- has the loop stopped paying for itself (PANDA+ decline
                       or match-rate plateau)?

These are deliberately separate: the fallback changes HOW the next round
trains, while the stop check ends the loop. A round can trip one without the
other.

Why the fallback matters even more after the ISUP-informed single-loss redesign:
Rules 1-3 rewrite flagged pixels at FULL weight inside seg_target (direct target
edit -- NOT the old 0.70·mask + 0.30·pseudo diluted blend). Un-flagged pixels
still come from the base target (Round 2+: previous model's raw argmax). Any
broad G3→G4 bias that Rules 1-3 did not catch is therefore taught at full
strength. PANDA+ is the only external check that can catch that drift -- do
not remove this gate. Leak-ratio tolerance was tightened from 10% → 5% for
the same reason (see G3_G4_LEAK_TOLERANCE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ROUNDS = 3
MATCH_RATE_PLATEAU_THRESHOLD = 0.03  # 3 percentage points
# Leak-ratio worsening cutoff. Was 1.10 (10%) under the old 0.70/0.30 dual-loss
# design, where a bad correction was diluted by the mask's majority vote. With
# full-weight ISUP target edits, the same bias is taught harder -- tighten to
# 5% so the gate trips earlier. cancer_dice / g5_dice still trip on ANY decline.
G3_G4_LEAK_TOLERANCE = 1.05

SEG_TARGET_MASK = "original_mask"
SEG_TARGET_MODEL = "model_prediction"


@dataclass(frozen=True)
class PandaPlusMetrics:
    """The external-validation numbers each round is judged on."""

    cancer_dice: float
    g5_dice: float
    g3_to_g4_leak_ratio: float


def bias_too_heavy(
    current: PandaPlusMetrics,
    previous: PandaPlusMetrics,
    *,
    leak_tolerance: float = G3_G4_LEAK_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Has the model-prediction base seg_target started doing real harm on PANDA+?

    Returns (tripped, reasons). Reasons are human-readable and meant to be
    printed prominently -- a silent fallback would be indistinguishable from
    the loop working as intended.

    Rules 1-3 only rewrite specifically flagged pixels. They cannot correct a
    broad, un-flagged bias in the previous round's raw predictions (the Round
    2+ base target). With full-weight ISUP edits, that uncorrected remainder is
    taught even harder -- PANDA+ is mandatory, not advisory.
    """
    reasons: list[str] = []
    if current.cancer_dice < previous.cancer_dice:
        reasons.append(
            f"cancer_dice declined {previous.cancer_dice:.4f} -> {current.cancer_dice:.4f}"
        )
    if current.g5_dice < previous.g5_dice:
        reasons.append(f"g5_dice declined {previous.g5_dice:.4f} -> {current.g5_dice:.4f}")
    leak_limit = previous.g3_to_g4_leak_ratio * leak_tolerance
    if current.g3_to_g4_leak_ratio > leak_limit:
        reasons.append(
            f"g3->g4 leak ratio worsened {previous.g3_to_g4_leak_ratio:.3f} -> "
            f"{current.g3_to_g4_leak_ratio:.3f} (limit {leak_limit:.3f})"
        )
    return bool(reasons), reasons


def apply_bias_fallback(
    seg_target_mode: str,
    current: PandaPlusMetrics,
    previous: PandaPlusMetrics,
) -> tuple[str, list[str]]:
    """Fold one round's result into the base-seg_target state machine.

    Once the fallback trips, the mode stays ``original_mask`` for the rest of
    the run -- there is no path back to ``model_prediction``, even if a later
    round looks healthy again. Operationally that means the next round's
    ``--seg-target-dir`` is omitted so training again uses the original mask
    as the base (Rules 1-3 may still rewrite flagged pixels on top).
    """
    if seg_target_mode == SEG_TARGET_MASK:
        return SEG_TARGET_MASK, []
    tripped, reasons = bias_too_heavy(current, previous)
    if tripped:
        return SEG_TARGET_MASK, reasons
    return SEG_TARGET_MODEL, []


def should_stop(
    current: PandaPlusMetrics,
    previous: PandaPlusMetrics,
    current_match_rate: float,
    previous_match_rate: float,
    *,
    plateau_threshold: float = MATCH_RATE_PLATEAU_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Stop the loop on PANDA+ decline or a match-rate plateau."""
    reasons: list[str] = []
    if current.cancer_dice < previous.cancer_dice:
        reasons.append(
            f"PANDA+ cancer_dice declined {previous.cancer_dice:.4f} -> {current.cancer_dice:.4f}"
        )
    if current.g5_dice < previous.g5_dice:
        reasons.append(
            f"PANDA+ g5_dice declined {previous.g5_dice:.4f} -> {current.g5_dice:.4f}"
        )
    gain = current_match_rate - previous_match_rate
    if gain < plateau_threshold:
        reasons.append(
            f"match rate plateaued: +{gain:.3f} < {plateau_threshold:.3f} "
            f"({previous_match_rate:.1%} -> {current_match_rate:.1%})"
        )
    return bool(reasons), reasons


def g3_to_g4_leak_ratio_from_confusion(confusion) -> float:
    """G3→G4 / G4→G3 count ratio from an NxN confusion (rows=true, cols=pred).

    Values > 1 mean the model over-calls G4 on true G3 more than the reverse --
    the known directional bias this project tracks on PANDA+.
    """
    cm = np.asarray(confusion)
    g3_to_g4 = float(cm[3, 4]) if cm.shape[0] > 4 else 0.0
    g4_to_g3 = float(cm[4, 3]) if cm.shape[0] > 4 else 0.0
    return g3_to_g4 / max(g4_to_g3, 1.0)
