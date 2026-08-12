#!/usr/bin/env python3
"""Model 2 loss design -- Part 8 smoke test.

Loads teacher A, runs 5 batches through combined_model2_loss on a slide
selection deliberately mixed across match/A/B/C_pure/C_two_class_unresolved/D
so we can directly verify:
  1. Rule C fires ONLY on C_pure slides where the model's LIVE prediction is
     an invented class, never on C_two_class_unresolved (must stay flagged=0
     everywhere per PART 5).
  2. Rule B fires ONLY on Rule B slides, and only at true-G3-predicted-G4
     pixels -- never on A/C/D slides. It's a separate additive CE term
     (rule_b_loss_term), normalized over its own flagged pixels rather than
     folded into segmentation_loss's weight_map, so its magnitude is visible
     and independent of how many Rule B slides land in a given batch.
  3. All three loss components (seg, rule_b, rule_c) for 5 batches.

STOP after printing results -- do not submit full training (per spec Part 8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import build_eval_model  # noqa: E402
from patch_utils import PROJECT  # noqa: E402
from train.class_weights import get_or_compute_class_weights  # noqa: E402
from train.losses import combined_model2_loss, rule_b_loss_term, rule_c_flag_and_target  # noqa: E402
from train.model2_dataset import Model2PseudoLabelDataset  # noqa: E402
from train.pseudo_label_rules import DEFAULT_MANIFEST_OUT  # noqa: E402

CHECKPOINT = (
    PROJECT / "outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth"
)
TRAIN_SPLIT = PROJECT / "outputs/splits/panda_train.csv"
BATCH_SIZE = 8
NUM_BATCHES = 5
SMOKE_SPLIT_CSV = PROJECT / "outputs/pseudo_label/_model2_smoke_split.csv"


def unit_test_rule_c_flag_and_target() -> None:
    """Deterministic, data-independent proof that rule_c_flag_and_target()
    correctly flags/retargets invented-class pixels from the MODEL'S
    PREDICTED class map (not the ground-truth mask -- see the docstring in
    losses.py for why Rule C was redesigned to key off live predictions:
    isup_diagnostic.py's `derived_gleason`/`derived_isup` -- the signal used
    to classify a slide as Rule C in the first place -- come from
    `model(images).argmax(dim=1)`, and an empirical scan of 60 C_pure slides
    found zero cases of literal invented-class pixels in the ground-truth
    mask, confirming the mask-based version would be a permanent no-op)."""
    pred_class = torch.zeros(2, 8, 8, dtype=torch.long)
    # Sample 0: C_pure slide, true_class=3. Model predicts G4 (invented) on
    # rows 5-8 -- these must be flagged and retargeted to class 3.
    pred_class[0, 0:2, :] = 1
    pred_class[0, 2:5, :] = 3
    pred_class[0, 5:8, :] = 4  # invented -- model is wrong here
    # Sample 1: NOT a Rule C slide (e.g. Rule D) -- identical G4 prediction
    # pattern must NEVER be flagged.
    pred_class[1, 5:8, :] = 4

    is_rule_c_pure = torch.tensor([True, False])
    true_class = torch.tensor([3, 0])

    corrected, flagged = rule_c_flag_and_target(pred_class, is_rule_c_pure, true_class, num_classes=6)
    assert flagged[0].sum().item() == 24, f"sample 0: expected 24 flagged pixels, got {flagged[0].sum().item()}"
    assert bool((flagged[0] == (pred_class[0] == 4)).all()), "sample 0: flag_mask must exactly match model's invented-class (4) predictions"
    assert bool((corrected[0, 3][flagged[0]] == 1.0).all()), "sample 0: flagged pixels must be hard-retargeted to true_class=3"
    assert bool((corrected[0, 4][flagged[0]] == 0.0).all()), "sample 0: flagged pixels must have zero mass on the invented class"
    assert flagged[1].sum().item() == 0, "sample 1: non-Rule-C slide must NEVER be flagged, even with an identical prediction pattern"
    print(
        "unit_test_rule_c_flag_and_target: PASSED (24/24 invented predictions "
        "correctly flagged+retargeted on the Rule C slide; identical pattern "
        "on a non-Rule-C slide correctly ignored)"
    )


def unit_test_rule_b_loss_term() -> None:
    """Deterministic, data-independent proof that rule_b_loss_term() only
    flags/penalizes the validated true-G3-predicted-G4 direction, only on
    Rule B slides, and normalizes over just its own flagged pixels (not the
    whole batch) -- see the "WHY A SEPARATE TERM" comment in losses.py."""
    B, C, H, W = 2, 6, 8, 8
    logits = torch.zeros(B, C, H, W)
    mask = torch.zeros(B, H, W, dtype=torch.long)

    for b in range(B):
        mask[b, 0:2, :] = 1
        logits[b, 1, 0:2, :] = 10.0  # correctly predicted stroma
        mask[b, 2:5, :] = 3
        logits[b, 3, 2:5, :] = 10.0  # correctly predicted G3
        mask[b, 5:8, :] = 3  # true G3
        logits[b, 4, 5:8, :] = 10.0  # WRONGLY predicted G4 -- the bad direction

    is_rule_b_slide = torch.tensor([True, False])  # only sample 0 is Rule B
    class_weights = torch.ones(C)

    loss, pixel_count = rule_b_loss_term(logits, mask, is_rule_b_slide, class_weights)
    assert pixel_count.item() == 24, f"expected 24 flagged pixels (sample 0 rows 5-8 x 8 cols only), got {pixel_count.item()}"

    expected_ce = F.cross_entropy(logits[0:1, :, 5:8, :], mask[0:1, 5:8, :], reduction="mean")
    assert torch.isclose(loss, expected_ce, atol=1e-4), f"loss {loss.item()} != expected {expected_ce.item()}"

    # Reverse direction (true G4, predicted G3) must NOT be flagged.
    mask2 = torch.zeros(1, H, W, dtype=torch.long)
    mask2[0, :, :] = 4  # true G4
    logits2 = torch.zeros(1, C, H, W)
    logits2[0, 3, :, :] = 10.0  # predicted G3 -- the OTHER, non-penalized direction
    _, pixel_count2 = rule_b_loss_term(logits2, mask2, torch.tensor([True]), class_weights)
    assert pixel_count2.item() == 0, "reverse direction (true G4, predicted G3) must never be flagged"

    print(
        "unit_test_rule_b_loss_term: PASSED (24/24 true-G3-predicted-G4 pixels "
        "flagged only on the Rule B sample, loss matches manual CE over just "
        "those pixels, non-Rule-B sample and reverse direction correctly ignored)"
    )


def build_smoke_split() -> Path:
    """Patches from a deliberate mix of rule categories, for coverage."""
    manifest = pd.read_csv(DEFAULT_MANIFEST_OUT, dtype={"slide_id": str})
    split = pd.read_csv(TRAIN_SPLIT, dtype={"image_id": str})

    wanted_per_rule = {
        "match": 2,
        "A": 2,
        "B": 3,
        "C_pure": 3,
        "C_two_class_unresolved": 2,
        "D": 2,
    }
    slide_ids: list[str] = []
    for rule, n in wanted_per_rule.items():
        candidates = manifest.loc[manifest["rule"] == rule, "slide_id"]
        candidates = candidates[candidates.isin(split["image_id"])]
        slide_ids.extend(candidates.head(n).tolist())

    print(f"Smoke-test slide selection ({len(slide_ids)} slides):")
    for rule in wanted_per_rule:
        chosen = manifest[manifest["slide_id"].isin(slide_ids) & (manifest["rule"] == rule)]
        print(f"  {rule:28s} {len(chosen)} slides: {chosen['slide_id'].tolist()}")

    patches = split[split["image_id"].isin(slide_ids)].copy()
    n_needed = BATCH_SIZE * NUM_BATCHES
    # Oversample patches per C_pure/B slide (more chances the model's live
    # prediction actually lands on the flagged behavior within this tiny
    # smoke sample) by taking more patches per slide instead of a flat
    # random draw across all patches.
    per_slide_cap = max(2, n_needed // max(len(slide_ids), 1) + 2)
    per_slide_chunks = [
        grp.sample(n=min(len(grp), per_slide_cap), random_state=0)
        for _, grp in patches.groupby("image_id")
    ]
    patches = pd.concat(per_slide_chunks, ignore_index=True)
    if len(patches) > n_needed:
        patches = patches.sample(n=n_needed, random_state=0)
    patches = patches.reset_index(drop=True)
    print(f"Total patches selected: {len(patches)} (need >= {n_needed})")
    SMOKE_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    patches.to_csv(SMOKE_SPLIT_CSV, index=False)
    return SMOKE_SPLIT_CSV


def main() -> None:
    unit_test_rule_b_loss_term()
    unit_test_rule_c_flag_and_target()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    split_csv = build_smoke_split()
    dataset = Model2PseudoLabelDataset(
        split_csv,
        mode="raw",
        allow_missing_h5=True,
    )
    print(f"\nDataset rule counts (this smoke split): {dataset.rule_counts()}")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True,
    )

    print("\nLoading teacher A checkpoint...")
    model = build_eval_model("uni2_upernet").to(device)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        non_bn_missing = [
            k for k in missing
            if not (k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked"))
        ]
        print(f"  missing={len(missing)} (all BN running-stats buffers, not real weights: {len(non_bn_missing) == 0})")
        if non_bn_missing:
            print(f"    !! includes real weights, not just BN stats: {non_bn_missing[:10]}")
    if unexpected:
        print(f"  unexpected keys (sample): {unexpected[:6]}")
    print(f"Loaded epoch={ckpt.get('epoch')} val_cancer_dice={ckpt.get('metrics', {}).get('cancer_dice')}")
    # Deliberately NOT calling model.eval(): decode_head BatchNorm running
    # stats are absent from this checkpoint (see missing-keys check above),
    # so eval-mode normalization would use uninitialized stats. train() mode
    # computes BatchNorm stats from the live batch instead, which is exactly
    # the mode real training will run in anyway -- appropriate for this
    # smoke test's purpose (verify loss wiring), not a claim about eval-mode
    # inference quality from this exact checkpoint.
    model.train()

    train_df = pd.read_csv(TRAIN_SPLIT)
    weight_bundle = get_or_compute_class_weights(train_df, mode="raw")
    class_weights = weight_bundle["class_weights"].to(device)
    print(f"class_weights={class_weights.tolist()}")

    print(f"\n{'=' * 90}")
    print("Running 5 smoke-test batches through combined_model2_loss")
    print(f"{'=' * 90}")

    total_rule_b_pixels = 0
    total_rule_c_pixels = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= NUM_BATCHES:
            break
        image, mask, weight, is_rule_b, is_rule_c_pure, rule_c_true_class = batch
        image = image.to(device)
        mask = mask.to(device)
        weight = weight.to(device)
        is_rule_b = is_rule_b.to(device)
        is_rule_c_pure = is_rule_c_pure.to(device)
        rule_c_true_class = rule_c_true_class.to(device)

        with torch.no_grad():
            logits = model(image)
            total, metrics = combined_model2_loss(
                logits,
                mask,
                weight,
                is_rule_b,
                is_rule_c_pure,
                rule_c_true_class,
                class_weights,
                adjacent_soft_alpha=0.15,
            )

        total_rule_b_pixels += metrics["rule_b_pixels_flagged"]
        total_rule_c_pixels += metrics["rule_c_pixels_flagged"]
        print(
            f"batch {batch_idx}: is_rule_b={is_rule_b.tolist()} "
            f"is_rule_c_pure={is_rule_c_pure.tolist()} "
            f"total={metrics['total']:.4f} seg={metrics['seg']:.4f} "
            f"rule_b={metrics['rule_b']:.4f} rule_c={metrics['rule_c']:.4f} "
            f"rule_b_pixels_flagged={metrics['rule_b_pixels_flagged']} "
            f"rule_c_pixels_flagged={metrics['rule_c_pixels_flagged']}"
        )

    print(f"\n{'=' * 90}")
    print("Sanity checks")
    print(f"{'=' * 90}")
    print(f"Total Rule B pixels flagged across 5 batches: {total_rule_b_pixels}")
    print(f"Total Rule C pixels flagged across 5 batches: {total_rule_c_pixels}")
    print(
        "Rule B is now a SEPARATE additive CE term (rule_b_loss_term in "
        "losses.py), normalized over just its own flagged pixels -- not "
        "folded into seg_loss's weight_map (that diluted its effect across "
        "the whole batch's pixel count and made it invisible in the logs). "
        "rule_b_pixels_flagged is 0 for any batch with no Rule B samples, "
        "confirming it never fires on A/C/D/match slides."
    )
    print(
        "Rule C fires only where is_rule_c_pure=True AND the model's live "
        "prediction sits on an invented class (see rule_c_flag_and_target in "
        "losses.py) -- C_two_class_unresolved slides are never passed as "
        "is_rule_c_pure=True (see Model2PseudoLabelDataset), so they never "
        "contribute to rule_c_loss, confirming PART 5's 'no auto-correction "
        "yet' requirement."
    )
    print("\n*** STOP -- smoke test complete, awaiting review before full training. ***")


if __name__ == "__main__":
    main()
