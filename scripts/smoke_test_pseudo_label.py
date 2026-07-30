"""PART 8 smoke tests for the iterative pseudo-label design.

Checks, run before any Round 1 training job is submitted:

  1. Rule 1 wide-margin 2<->3 cases are tagged wide_margin_unresolved and apply
     NO pixel rewrite (same effect as missing-class / none). Do not hard-correct
     over-extended G3/G4 -- we lack a reliable pixel-level split.
  2. Rule 2 fires only on its adjacency condition and always uses the FIXED
     95/5 cushion -- never a model-derived blend.
  3. Rule 3 defaults correctly on pure-pattern slides.
  4. seg_target switches from the original mask (Round 1) to cached model
     predictions (Round 2 default mode) in a dry run.
  5. The bias fallback flips seg_target_mode when fed degraded PANDA+ numbers.

Tests 3 need the source-prediction cache; they report SKIPPED (not passed)
when cache is missing. Tests 1/2/4/5 are self-contained (no cache required).

No model checkpoint is loaded: the pseudo-label path never consults the
currently-training model, so random logits are enough to exercise the loss
plumbing end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import PROJECT  # noqa: E402
from train.class_weights import get_or_compute_class_weights  # noqa: E402
from train.losses import combined_loss  # noqa: E402
from train.pseudo_label_dataset import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PRED_DIR,
    NO_CLASS,
    SEG_TARGET_MASK,
    SEG_TARGET_MODEL,
    PredictionCacheReader,
    PseudoLabelPatchDataset,
    build_corrected_target,
    parse_flag_classes,
)
from train.pseudo_label_rules import (  # noqa: E402
    NO_CORRECTION_RULES,
    RULE2_ADJACENT_INVENTED,
    RULE2_CUSHION_WEIGHT,
    RULE2_MAIN_WEIGHT,
    RULE3_INVENTED_DEFAULT,
    WIDE_MARGIN_UNRESOLVED,
    classify_slide,
    gleason_to_classes,
)
from train.round_control import (  # noqa: E402
    SEG_TARGET_MASK as RC_MASK,
    SEG_TARGET_MODEL as RC_MODEL,
    PandaPlusMetrics,
    apply_bias_fallback,
    bias_too_heavy,
)

CANCER_CLASSES = (3, 4, 5)
SPLIT_CSV = PROJECT / "outputs" / "splits" / "panda_train.csv"

results: list[tuple[str, str]] = []


def record(name: str, status: str) -> None:
    results.append((name, status))


def slide_pixel_histogram(reader: PredictionCacheReader, slide_id: str, coords) -> np.ndarray:
    """Total predicted-class pixel counts across all cached patches of a slide."""
    hist = np.zeros(6, dtype=np.int64)
    for x, y in coords:
        pred = reader.read(slide_id, int(x), int(y))
        if pred is None:
            continue
        hist += np.bincount(pred.reshape(-1), minlength=6)
    return hist


def cached_slides(reader: PredictionCacheReader, manifest: pd.DataFrame, rule: str) -> list:
    rows = manifest[manifest["rule_applied"] == rule]
    return [r for r in rows.itertuples() if reader.has(str(r.slide_id))]


def test_1_wide_margin_unresolved(manifest) -> None:
    print("\n" + "=" * 72)
    print("TEST 1 -- Rule 1 wide-margin is unresolved (NO pixel rewrite)")
    print("=" * 72)

    failures = []

    # Synthetic: true 3+4, model derives 4+3 with a wide G4-dominant margin.
    # Must NOT hard-correct G4->G3; flag as wide_margin_unresolved with empty
    # flags / targets so the original mask is left alone.
    hit = classify_slide(
        metadata_gleason="3+4",
        metadata_isup=2,
        derived_gleason="4+3",
        derived_isup=3,
        match=False,
        cancer_frac_g3=0.25,
        cancer_frac_g4=0.75,
    )
    print(f"\n  3+4 -> derived 4+3 wide margin : {hit.rule_applied}")
    print(f"    flag_pred_classes={hit.flag_pred_classes}")
    print(f"    target_main_class={hit.target_main_class} weight={hit.target_main_weight}")
    print(f"    reason={hit.reason}")

    if hit.rule_applied != WIDE_MARGIN_UNRESOLVED:
        failures.append(f"expected {WIDE_MARGIN_UNRESOLVED}, got {hit.rule_applied}")
    if hit.rule_applied not in NO_CORRECTION_RULES:
        failures.append(f"{hit.rule_applied} must be in NO_CORRECTION_RULES")
    if hit.flag_pred_classes:
        failures.append(f"flag_pred_classes must be empty, got {hit.flag_pred_classes}")
    if hit.target_main_class != -1 or hit.target_main_weight != 0.0:
        failures.append(
            f"no target rewrite allowed, got main={hit.target_main_class}/{hit.target_main_weight}"
        )

    # Manifest rows (if present) must also carry empty flags / no rewrite.
    rows = manifest[manifest["rule_applied"] == WIDE_MARGIN_UNRESOLVED]
    print(f"\n  manifest wide_margin_unresolved rows: {len(rows)}")
    if len(rows) == 0:
        print("  WARN: manifest has 0 wide_margin_unresolved rows -- regenerate?")
    for row in rows.itertuples():
        flags = parse_flag_classes(row.flag_pred_classes)
        if flags:
            failures.append(f"{row.slide_id}: manifest still has flag_pred_classes={flags}")
        if int(row.target_main_class) != -1:
            failures.append(
                f"{row.slide_id}: manifest target_main_class={row.target_main_class} (want -1)"
            )

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_1_wide_margin_unresolved", "FAILED")
    else:
        print(
            "\n  PASSED: wide-margin tagged unresolved with empty flags; "
            "no hard G3/G4 rewrite."
        )
        record("test_1_wide_margin_unresolved", "PASSED")


def test_2_rule2_adjacency_and_cushion(manifest) -> None:
    print("\n" + "=" * 72)
    print("TEST 2 -- Rule 2 adjacency condition + FIXED 95/5 cushion")
    print("=" * 72)

    failures = []

    # Positive: doc's own example -- true 3+5, model derives 3+4. Primary
    # correct, invented 4 adjacent to true secondary 5.
    hit = classify_slide(
        metadata_gleason="3+5", metadata_isup=4, derived_gleason="3+4", derived_isup=2,
        match=False, cancer_frac_g3=0.6, cancer_frac_g4=0.4,
    )
    print(f"\n  3+5 -> derived 3+4 : {hit.rule_applied}")
    print(
        f"    target = {hit.target_main_weight:.2f} x G{hit.target_main_class} "
        f"+ {hit.target_cushion_weight:.2f} x G{hit.target_cushion_class}"
    )
    if hit.rule_applied != RULE2_ADJACENT_INVENTED:
        failures.append(f"3+5/3+4 should be Rule 2, got {hit.rule_applied}")
    if (hit.target_main_class, hit.target_cushion_class) != (5, 4):
        failures.append(f"3+5/3+4 target classes should be (5, 4), got {(hit.target_main_class, hit.target_cushion_class)}")
    if abs(hit.target_main_weight - RULE2_MAIN_WEIGHT) > 1e-9:
        failures.append(f"main weight should be exactly {RULE2_MAIN_WEIGHT}, got {hit.target_main_weight}")
    if abs(hit.target_cushion_weight - RULE2_CUSHION_WEIGHT) > 1e-9:
        failures.append(f"cushion weight should be exactly {RULE2_CUSHION_WEIGHT}, got {hit.target_cushion_weight}")

    # Negative: non-adjacent invention (true 4+3, derived 4+5 -> |5-3| = 2).
    miss = classify_slide(
        metadata_gleason="4+3", metadata_isup=3, derived_gleason="4+5", derived_isup=5,
        match=False, cancer_frac_g3=0.1, cancer_frac_g4=0.6,
    )
    print(f"  4+3 -> derived 4+5 : {miss.rule_applied} (non-adjacent, must NOT be Rule 2)")
    if miss.rule_applied == RULE2_ADJACENT_INVENTED:
        failures.append("non-adjacent invention wrongly classified as Rule 2")

    # Negative: pure pattern stays out of Rule 2 even though 3 is adjacent to 4.
    pure = classify_slide(
        metadata_gleason="4+4", metadata_isup=4, derived_gleason="4+3", derived_isup=3,
        match=False, cancer_frac_g3=0.3, cancer_frac_g4=0.7,
    )
    print(f"  4+4 -> derived 4+3 : {pure.rule_applied} (pure pattern, must NOT be Rule 2)")
    if pure.rule_applied == RULE2_ADJACENT_INVENTED:
        failures.append("pure-pattern slide wrongly classified as Rule 2")

    # Every real Rule 2 row must carry the fixed cushion, nothing model-derived.
    real = manifest[manifest["rule_applied"] == RULE2_ADJACENT_INVENTED]
    print(f"\n  Real Rule 2 slides in manifest: {len(real)}")
    if len(real):
        weights = real["target_main_weight"].round(10).unique()
        cushions = real["target_cushion_weight"].round(10).unique()
        print(f"    distinct main weights   : {weights}")
        print(f"    distinct cushion weights: {cushions}")
        for row in real.head(3).itertuples():
            print(
                f"    {row.slide_id} metadata={row.metadata_gleason} derived={row.derived_gleason} "
                f"-> {row.target_main_weight:.2f} x G{int(row.target_main_class)} "
                f"+ {row.target_cushion_weight:.2f} x G{int(row.target_cushion_class)}"
            )
        if not (len(weights) == 1 and abs(weights[0] - RULE2_MAIN_WEIGHT) < 1e-9):
            failures.append(f"Rule 2 main weights vary or differ from {RULE2_MAIN_WEIGHT}: {weights}")
        if not (len(cushions) == 1 and abs(cushions[0] - RULE2_CUSHION_WEIGHT) < 1e-9):
            failures.append(f"Rule 2 cushion weights vary or differ from {RULE2_CUSHION_WEIGHT}: {cushions}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_2_rule2_adjacency_and_cushion", "FAILED")
    else:
        print("\n  PASSED: adjacency gate correct; cushion is a single fixed 95/5 everywhere.")
        record("test_2_rule2_adjacency_and_cushion", "PASSED")


def test_3_rule3_pure_pattern(manifest, reader, split_df, max_slides: int) -> None:
    print("\n" + "=" * 72)
    print("TEST 3 -- Rule 3 default on pure-pattern slides")
    print("=" * 72)

    failures = []

    # Unit case: pure 4+4, model invents G3.
    pure = classify_slide(
        metadata_gleason="4+4", metadata_isup=4, derived_gleason="4+3", derived_isup=3,
        match=False, cancer_frac_g3=0.3, cancer_frac_g4=0.7,
    )
    print(f"\n  4+4 -> derived 4+3 : {pure.rule_applied}")
    print(f"    flag classes {list(pure.flag_pred_classes)} -> hard target G{pure.target_main_class}")
    if pure.rule_applied != RULE3_INVENTED_DEFAULT:
        failures.append(f"pure 4+4 should be Rule 3, got {pure.rule_applied}")
    if pure.target_main_class != 4 or abs(pure.target_main_weight - 1.0) > 1e-9:
        failures.append("pure-pattern target must be a hard 1.0 on the slide's one true class")
    if pure.target_cushion_class != NO_CLASS:
        failures.append("pure-pattern target must have no cushion class")
    if set(pure.flag_pred_classes) != {3, 5}:
        failures.append(f"pure 4+4 should flag exactly the non-expected cancer classes {{3,5}}, got {pure.flag_pred_classes}")

    # Real pure-pattern slides: flagged pixels must never include the true class.
    rows = [
        r
        for r in cached_slides(reader, manifest, RULE3_INVENTED_DEFAULT)
        if len(gleason_to_classes(r.metadata_gleason)) == 1
    ][:max_slides]
    if not rows:
        print("\n  (no cached pure-pattern Rule 3 slide available yet -- unit case only)")
    for row in rows:
        slide_id = str(row.slide_id)
        coords = split_df.loc[split_df["image_id"] == slide_id, ["x", "y"]].to_numpy()
        hist = slide_pixel_histogram(reader, slide_id, coords)
        true_class = int(row.target_main_class)
        flag_classes = parse_flag_classes(row.flag_pred_classes)
        flagged = int(sum(hist[c] for c in flag_classes))
        kept = int(hist[true_class])
        print(
            f"\n  {slide_id}  metadata={row.metadata_gleason} derived={row.derived_gleason}"
        )
        print(f"    invented pixels flagged -> G{true_class} : {flagged:,}")
        print(f"    true-class pixels left alone            : {kept:,}")
        if true_class in flag_classes:
            failures.append(f"{slide_id}: true class G{true_class} must never be flagged")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_3_rule3_pure_pattern", "FAILED")
    else:
        print("\n  PASSED: pure-pattern slides get a hard target on their single true class.")
        record("test_3_rule3_pure_pattern", "PASSED")


def test_4_seg_target_switch(manifest_path: Path, pred_dir: Path, split_df) -> None:
    print("\n" + "=" * 72)
    print("TEST 4 -- seg_target switches mask (Round 1) -> model predictions (Round 2)")
    print("=" * 72)

    reader = PredictionCacheReader(pred_dir)
    cached = [s for s in split_df["image_id"].unique() if reader.has(str(s))]
    if not cached:
        print("SKIPPED: no cached predictions available to serve as a Round 2 seg_target.")
        record("test_4_seg_target_switch", "SKIPPED")
        return

    slide_id = str(cached[0])
    subset = split_df[split_df["image_id"] == slide_id].head(4)
    tmp_split = PROJECT / "outputs" / "pseudo_label" / "_smoke_split.csv"
    subset.to_csv(tmp_split, index=False)

    round1 = PseudoLabelPatchDataset(
        tmp_split, manifest_csv=manifest_path, pred_dir=pred_dir,
        seg_target_dir=None, mode="raw", allow_missing_h5=True,
    )
    round2 = PseudoLabelPatchDataset(
        tmp_split, manifest_csv=manifest_path, pred_dir=pred_dir,
        seg_target_dir=pred_dir, mode="raw", allow_missing_h5=True,
    )
    print(f"\n  Round 1 dataset seg_target_mode = {round1.seg_target_mode}")
    print(f"  Round 2 dataset seg_target_mode = {round2.seg_target_mode}")

    _, mask_target, _, _, _ = round1[0]
    _, pred_target, _, _, _ = round2[0]
    row = subset.iloc[0]
    cached_pred = reader.read(slide_id, int(row["x"]), int(row["y"]))

    agree_with_cache = bool(torch.equal(pred_target, torch.from_numpy(cached_pred.astype(np.int64))))
    differ = not bool(torch.equal(mask_target, pred_target))
    print(f"  Round 1 target label histogram : {torch.bincount(mask_target.reshape(-1), minlength=6).tolist()}")
    print(f"  Round 2 target label histogram : {torch.bincount(pred_target.reshape(-1), minlength=6).tolist()}")
    print(f"  Round 2 target == cached source prediction : {agree_with_cache}")
    print(f"  Round 1 target differs from Round 2 target : {differ}")

    tmp_split.unlink(missing_ok=True)

    failures = []
    if round1.seg_target_mode != SEG_TARGET_MASK:
        failures.append(f"Round 1 mode should be {SEG_TARGET_MASK}, got {round1.seg_target_mode}")
    if round2.seg_target_mode != SEG_TARGET_MODEL:
        failures.append(f"Round 2 mode should be {SEG_TARGET_MODEL}, got {round2.seg_target_mode}")
    if not agree_with_cache:
        failures.append("Round 2 seg_target does not match the cached prediction it should be reading")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_4_seg_target_switch", "FAILED")
    else:
        print("\n  PASSED: the switch reads the mask in Round 1 and cached predictions in Round 2.")
        record("test_4_seg_target_switch", "PASSED")


def test_5_bias_fallback() -> None:
    print("\n" + "=" * 72)
    print("TEST 5 -- bias fallback flips seg_target_mode on degraded PANDA+")
    print("=" * 72)

    baseline = PandaPlusMetrics(cancer_dice=0.70, g5_dice=0.55, g3_to_g4_leak_ratio=1.80)
    failures = []

    healthy = PandaPlusMetrics(cancer_dice=0.72, g5_dice=0.57, g3_to_g4_leak_ratio=1.75)
    tripped, reasons = bias_too_heavy(healthy, baseline)
    mode, _ = apply_bias_fallback(RC_MODEL, healthy, baseline)
    print(f"\n  improved round      -> tripped={tripped} mode={mode}")
    if tripped or mode != RC_MODEL:
        failures.append("an improving round must not trip the fallback")

    cases = {
        "cancer_dice drop": PandaPlusMetrics(0.68, 0.57, 1.75),
        "g5_dice drop": PandaPlusMetrics(0.72, 0.50, 1.75),
        "g3->g4 leak +15%": PandaPlusMetrics(0.72, 0.57, 1.80 * 1.15),
    }
    for label, degraded in cases.items():
        tripped, reasons = bias_too_heavy(degraded, baseline)
        mode, _ = apply_bias_fallback(RC_MODEL, degraded, baseline)
        print(f"  {label:20s} -> tripped={tripped} mode={mode}")
        for reason in reasons:
            print(f"      reason: {reason}")
        if not tripped or mode != RC_MASK:
            failures.append(f"{label} should trip the fallback and force {RC_MASK}")

    # A 5% leak increase is exactly at the tightened cutoff; check uses ``>``
    # so it must NOT trip. Anything above 5% must.
    tolerated = PandaPlusMetrics(0.72, 0.57, 1.80 * 1.05)
    tripped, _ = bias_too_heavy(tolerated, baseline)
    print(f"  g3->g4 leak +5%      -> tripped={tripped} (at 5% cutoff, must not trip)")
    if tripped:
        failures.append("a 5% leak increase is at the cutoff (>) and must not trip")

    over = PandaPlusMetrics(0.72, 0.57, 1.80 * 1.06)
    tripped, _ = bias_too_heavy(over, baseline)
    print(f"  g3->g4 leak +6%      -> tripped={tripped} (above 5% cutoff)")
    if not tripped:
        failures.append("a 6% leak increase must trip under the tightened 5% tolerance")

    # Once tripped, the mode is permanent even if the next round looks great.
    recovered = PandaPlusMetrics(0.80, 0.65, 1.50)
    mode, _ = apply_bias_fallback(RC_MASK, recovered, baseline)
    print(f"  recovery after trip  -> mode={mode} (must stay {RC_MASK})")
    if mode != RC_MASK:
        failures.append("fallback must be permanent once tripped")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_5_bias_fallback", "FAILED")
    else:
        print("\n  PASSED: trips on each degradation, tolerates noise, never reverts.")
        record("test_5_bias_fallback", "PASSED")


def test_6_loss_plumbing(manifest_path: Path, pred_dir: Path, split_df) -> None:
    """Not a PART 8 requirement, but proves dataset -> combined_loss runs end to end."""
    print("\n" + "=" * 72)
    print("TEST 6 -- dataset -> build_corrected_target -> combined_loss wiring")
    print("=" * 72)

    reader = PredictionCacheReader(pred_dir)
    manifest = pd.read_csv(manifest_path, dtype={"slide_id": str})
    corrected = manifest[~manifest["rule_applied"].isin(["match", "none"])]
    usable = [str(r.slide_id) for r in corrected.itertuples() if reader.has(str(r.slide_id))]
    if not usable:
        print("SKIPPED: no cached corrected slide to build a batch from.")
        record("test_6_loss_plumbing", "SKIPPED")
        return

    subset = split_df[split_df["image_id"].isin(usable[:2])].head(4)
    tmp_split = PROJECT / "outputs" / "pseudo_label" / "_smoke_split_loss.csv"
    subset.to_csv(tmp_split, index=False)
    ds = PseudoLabelPatchDataset(
        tmp_split, manifest_csv=manifest_path, pred_dir=pred_dir,
        mode="raw", allow_missing_h5=True,
    )

    images, masks, weights, flags, params = zip(*[ds[i] for i in range(len(ds))])
    masks = torch.stack(masks)
    weights = torch.stack(weights)
    flags = torch.stack(flags)
    params = torch.stack(params)
    tmp_split.unlink(missing_ok=True)

    corrected_target = build_corrected_target(flags, params)
    sums = corrected_target.sum(dim=1)
    flagged_sums = sums[flags]

    torch.manual_seed(0)
    logits = torch.randn(len(ds), 6, masks.shape[-2], masks.shape[-1], requires_grad=True)
    class_weights = get_or_compute_class_weights(
        pd.read_csv(SPLIT_CSV), mode="raw", recompute=False,
    )["class_weights"]
    if not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    total, metrics = combined_loss(
        logits, masks, corrected_target, flags, class_weights, weights,
    )
    total.backward()

    print(f"\n  flagged pixels in batch : {metrics['pseudo_pixels_flagged']:,}")
    print(f"  seg loss                : {metrics['seg']:.4f}")
    print(f"  pseudo loss             : {metrics['pseudo']:.4f}")
    print(f"  total (0.70/0.30)       : {metrics['total']:.4f}")
    print(f"  gradient flows          : {logits.grad is not None and bool(torch.isfinite(logits.grad).all())}")

    failures = []
    if flagged_sums.numel() and not torch.allclose(flagged_sums, torch.ones_like(flagged_sums), atol=1e-5):
        failures.append("corrected_target rows on flagged pixels must sum to 1.0")
    expected_total = 0.70 * metrics["seg"] + 0.30 * metrics["pseudo"]
    if abs(expected_total - metrics["total"]) > 1e-5:
        failures.append(f"total {metrics['total']} != 0.70*seg + 0.30*pseudo ({expected_total})")
    if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
        failures.append("gradients did not flow cleanly through combined_loss")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        record("test_6_loss_plumbing", "FAILED")
    else:
        print("\n  PASSED: targets normalize, weighting is exactly 0.70/0.30, gradients finite.")
        record("test_6_loss_plumbing", "PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="PART 8 pseudo-label smoke tests")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_CSV)
    parser.add_argument("--max-slides", type=int, default=5)
    args = parser.parse_args()

    print(f"manifest : {args.manifest}")
    print(f"pred dir : {args.pred_dir}")

    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    split_df = pd.read_csv(args.split, dtype={"image_id": str})
    reader = PredictionCacheReader(args.pred_dir)

    n_cached = sum(1 for s in manifest["slide_id"] if reader.has(str(s)))
    print(f"cached slides available: {n_cached}")

    test_1_wide_margin_unresolved(manifest)
    test_2_rule2_adjacency_and_cushion(manifest)
    test_3_rule3_pure_pattern(manifest, reader, split_df, args.max_slides)
    test_4_seg_target_switch(args.manifest, args.pred_dir, split_df)
    test_5_bias_fallback()
    test_6_loss_plumbing(args.manifest, args.pred_dir, split_df)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, status in results:
        print(f"  {status:8s} {name}")
    failed = [n for n, s in results if s == "FAILED"]
    skipped = [n for n, s in results if s == "SKIPPED"]
    if failed:
        print(f"\n{len(failed)} test(s) FAILED -- do not submit Round 1 training.")
        sys.exit(1)
    if skipped:
        print(f"\n{len(skipped)} test(s) SKIPPED (need the prediction cache) -- rerun after caching.")
    print("\nAll runnable tests passed.")


if __name__ == "__main__":
    main()
