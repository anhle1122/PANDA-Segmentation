# Round 1 Label-Generation Summary

Manual review gate. No training has been run and no mask file has been modified.

## Inputs

- Manifest: `/common/omarmlab/members/anh/panda_project/outputs/pseudo_label/round1_rule_manifest.csv`
- Source-prediction cache: `/common/omarmlab/members/anh/panda_project/outputs/pseudo_label/round1_source_pred`
- Split: `/common/omarmlab/members/anh/panda_project/outputs/splits/panda_train.csv` (440,371 patches / 3,746 slides)
- Corrected slides with cache scanned: 629 / 629

## Slide-level rule breakdown

| Rule | Slides | % of split |
| --- | ---: | ---: |
| `match` | 2,018 | 53.9% |
| `none` | 1,099 | 29.3% |
| `rule3_invented_default` | 371 | 9.9% |
| `rule1_wide_margin` | 187 | 5.0% |
| `rule1_soft_tie` | 62 | 1.7% |
| `rule2_adjacent_invented` | 9 | 0.2% |
| **corrected total** | **629** | **16.8%** |

## Pixel-level impact (source-model predicted pixels rewritten)

| Rule | Slides | Patches | Flagged pixels | % of those slides' cancer pixels | % of their total pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rule1_soft_tie` | 62 | 6,950 | 936,094,515 | 97.4% | 51.4% |
| `rule1_wide_margin` | 187 | 22,575 | 1,451,794,001 | 46.5% | 24.5% |
| `rule2_adjacent_invented` | 9 | 1,035 | 99,694,920 | 66.6% | 36.7% |
| `rule3_invented_default` | 371 | 44,004 | 2,918,287,858 | 46.4% | 25.3% |
| **all rules** | **629** | **74,564** | **5,405,871,294** | **51.4%** | **27.7%** |

Across the whole training split those 5,405,871,294 pixels are 4.68% of all pixels. Only they receive `pseudo_loss`; every other pixel is trained by `seg_loss` against the original mask.

## Worked examples

### `rule1_soft_tie` (62 slides)

Largest edits first.

| Slide | metadata | derived | flags pred | -> target | flagged px | % of slide cancer |
| --- | --- | --- | --- | --- | ---: | ---: |
| `e62a7175ca46` | 4+3 (ISUP 3) | 3+4 (ISUP 2) | G3, G4 | 0.52xG3 + 0.48xG4 | 28,743,699 | 97.1% |
| `389758c9e3aa` | 4+3 (ISUP 3) | 3+4 (ISUP 2) | G3, G4 | 0.52xG3 + 0.48xG4 | 23,544,483 | 98.1% |
| `a4edf7a1424c` | 4+3 (ISUP 3) | 3+4 (ISUP 2) | G3, G4 | 0.50xG3 + 0.50xG4 | 23,441,983 | 96.3% |
| `bd2e60126032` | 3+4 (ISUP 2) | 4+3 (ISUP 3) | G3, G4 | 0.43xG3 + 0.57xG4 | 23,323,931 | 97.3% |

### `rule1_wide_margin` (187 slides)

Largest edits first.

| Slide | metadata | derived | flags pred | -> target | flagged px | % of slide cancer |
| --- | --- | --- | --- | --- | ---: | ---: |
| `7dcb52d4a04f` | 3+4 (ISUP 2) | 4+3 (ISUP 3) | G4 | 1.00xG3 | 22,748,838 | 69.7% |
| `7df260d73a12` | 3+4 (ISUP 2) | 4+3 (ISUP 3) | G4 | 1.00xG3 | 22,289,173 | 67.8% |
| `c2306aeec8a2` | 3+4 (ISUP 2) | 4+3 (ISUP 3) | G4 | 1.00xG3 | 21,063,176 | 66.7% |
| `85498a995584` | 3+4 (ISUP 2) | 4+3 (ISUP 3) | G4 | 1.00xG3 | 18,053,178 | 70.4% |

### `rule2_adjacent_invented` (9 slides)

Largest edits first.

| Slide | metadata | derived | flags pred | -> target | flagged px | % of slide cancer |
| --- | --- | --- | --- | --- | ---: | ---: |
| `df8bd94723b9` | 3+5 (ISUP 4) | 3+4 (ISUP 2) | G4 | 0.95xG5 + 0.05xG4 | 20,413,106 | 68.6% |
| `fc115a1b3e83` | 5+3 (ISUP 4) | 5+4 (ISUP 5) | G4 | 0.95xG3 + 0.05xG4 | 18,986,585 | 64.5% |
| `88fcb6de1ead` | 3+5 (ISUP 4) | 3+4 (ISUP 2) | G4 | 0.95xG5 + 0.05xG4 | 16,868,552 | 69.5% |
| `3e5fc2e7be26` | 5+3 (ISUP 4) | 5+4 (ISUP 5) | G4 | 0.95xG3 + 0.05xG4 | 13,097,326 | 67.2% |

### `rule3_invented_default` (371 slides)

Largest edits first.

| Slide | metadata | derived | flags pred | -> target | flagged px | % of slide cancer |
| --- | --- | --- | --- | --- | ---: | ---: |
| `1bc342d4ebef` | 3+3 (ISUP 1) | 3+4 (ISUP 2) | G4, G5 | 1.00xG3 | 25,051,777 | 74.3% |
| `27edf875a120` | 3+3 (ISUP 1) | 3+4 (ISUP 2) | G4, G5 | 1.00xG3 | 24,983,313 | 75.6% |
| `88e8c1a2565d` | 3+3 (ISUP 1) | 3+4 (ISUP 2) | G4, G5 | 1.00xG3 | 24,310,196 | 73.0% |
| `fd08b2bce422` | 3+3 (ISUP 1) | 3+4 (ISUP 2) | G4, G5 | 1.00xG3 | 23,959,969 | 74.3% |

## Distribution of how much each slide is rewritten

| Rule | median % of cancer pixels flagged | p90 | max |
| --- | ---: | ---: | ---: |
| `rule1_soft_tie` | 97.6% | 98.7% | 99.4% |
| `rule1_wide_margin` | 35.2% | 70.8% | 77.2% |
| `rule2_adjacent_invented` | 67.1% | 69.5% | 69.7% |
| `rule3_invented_default` | 30.4% | 73.9% | 96.2% |

Slides where the rule flags **every** predicted cancer pixel: 0. On these the corrected target replaces the model's entire cancer prediction, so they are the highest-risk edits to review.

## Review item: corrections that erase a metadata-required class

A correction is flagged here when it rewrites a class the slide's own metadata Gleason says is present AND gives that class no mass in the target. Rule 1's near-tie branch and Rule 2 are excluded by construction: they keep the flagged class alive in the main or cushion slot, so they rebalance rather than erase. Example: metadata `3+4`, model derives `4+3`. Rule 1's wide-margin branch rewrites every predicted G4 pixel to G3, so the resulting `corrected_target` contains no G4 at all -- even though `3+4` means G4 is genuinely there as the secondary pattern. The correction fixes the primary/secondary ordering by removing the secondary rather than by rebalancing it.

Slides affected: **187** of 629 scanned.

| Rule | slides erasing a metadata-required class |
| --- | ---: |
| `rule1_wide_margin` | 187 |

| Slide | rule | metadata | derived | class erased | pixels |
| --- | --- | --- | --- | --- | ---: |
| `7dcb52d4a04f` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 22,748,838 |
| `7df260d73a12` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 22,289,173 |
| `c2306aeec8a2` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 21,063,176 |
| `85498a995584` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 18,053,178 |
| `b4bebe718759` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 17,221,193 |
| `7eb79f361942` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 17,139,196 |
| `7453f914420e` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 16,785,775 |
| `1d5d638e2552` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 16,645,967 |
| `eda6d9fc5d58` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 16,591,959 |
| `91af122d94f8` | `rule1_wide_margin` | 3+4 | 4+3 | G4 | 16,271,112 |

This follows the design doc's wide-margin rule as written. It is called out because it is the one place where a correction can contradict the metadata it is derived from. `seg_loss` (weight 0.70) still trains those pixels against the original mask, which retains the true G3/G4 mix, so the net signal is a nudge rather than an erasure -- but the `pseudo_loss` target itself is unambiguous.

## Loss configuration for Round 1

- `seg_target` = **original PANDA mask** (Round 1; nothing else exists yet)
- `w_seg` = 0.70, `w_pseudo` = 0.30 (fixed for every round)
- `pseudo_loss` is averaged over flagged pixels only
- Validation stays on the original mask with the plain segmentation loss, so
  val metrics remain comparable across rounds

