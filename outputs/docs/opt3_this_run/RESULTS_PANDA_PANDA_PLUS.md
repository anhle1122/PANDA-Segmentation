# PANDA / PANDA+ results (through live ep34)

**Date:** 2026-08-17  
**Judge:** prefer **PANDA+ cancer Dice**, then PANDA+ ISUP / G5 precision. PANDA val Dice is in-domain and can disagree.

## How to read the columns

| Column | What it is | n |
|--------|------------|--:|
| **PANDA val** | Train-log cancer Dice (20k-patch val subset) | 472 slides (Omar-6 grouped split) |
| **PANDA ISUP** | Model paint → ISUP vs clinical, val split, thr=0 | 472 slides |
| **PANDA+ Dice** | Cancer Dice on PANDA+, `gt≥2` labeled pixels only | 4,688 patches / 48 slides |
| **PANDA+ ISUP** | Gland/mask-derived ISUP, thr=0, pred on labeled pixels | 48 slides |

PANDA ISUP and PANDA+ ISUP are **not** the same test (clinical vs mask-derived; mixed 472 vs cancer-heavy 48).

`—` = not scored. Do not treat missing as zero.

---

## Leaderboard (PANDA+ cancer Dice, scored ckpts only)

| Rank | Run | Ep | PANDA val | PANDA+ Dice | PANDA+ ISUP | PANDA ISUP |
|-----:|-----|---:|----------:|------------:|------------:|-----------:|
| 1 | **Omar-6 live** `opt3_omar6_grouped_soft01` | **29** | 0.550 | **0.609** | **60.4%** | **47.0%** |
| 2 | Omar-6 live | 30 | 0.543 | 0.600 | 60.4% | 38.8% |
| 3 | Omar-6 live | 7 | **0.608** | 0.587 | 62.5% | — |
| 4 | Omar-6 live | 33 | 0.540 | 0.584 | 50.0% | 36.4% |
| 5 | Omar-6 live | 32 | 0.507 | 0.582 | 60.4% | 38.8% |
| 6 | Omar-6 live | 15 | 0.579 | 0.579 | 52.1% | — |
| 7 | Omar-6 live | 34 | 0.520 | 0.573 | 60.4% | 41.3% |
| 8 | Omar-6 live | 28 | 0.569 | 0.565 | 54.2% | 40.3% |
| 9 | **Teacher A** `h200x4` | 5 | 0.602 | **0.563** | — | — |
| 10 | Teacher A | 30 | 0.677 | 0.560 | — | — |
| 11 | Teacher A | 25 | 0.668 | 0.558 | — | — |
| 12 | Teacher A | 15 | 0.639 | 0.556 | — | — |
| 13 | Teacher A | 35 | 0.726 | 0.554 | — | — |
| 14 | Teacher A (usual cite) | **42** | **0.742** | 0.554 | *31.2% invalid* | 53.9% |
| 15 | Rules R4 g45soft | 6 | 0.645 | 0.554 | — | — |
| 16 | 10k g45soft | 72 | 0.716 | 0.552 | — | — |
| 17 | First Opt3 | 12 | 0.619 | 0.551 | — | 52.6% *train* |
| 18 | First Opt3 | 21 | **0.636** | 0.546 | 58.3%† | 57.3% *train* |

† First Opt3 PANDA+ ISUP used `min_area_pct=0.05`; Omar-6 live uses **0.0**.

---

## 1. This run — Omar-6 live (`opt3_omar6_grouped_soft01`)

Job **5443101**, 2×H200, grouped split, λ_slide warmup → **0.3** from ep10, λ_grade=0.3.  
Old in-memory trainer: LoRA wrapped but **not in AdamW**; ISUP live not chunk-checkpointed. OOMd ep37 (~131G). Named ckpts through ep36 still on disk.

**Best external:** ep29. **Best PANDA val:** ep7.

### Scored external (plus ep7 / ep15)

| Ep | PANDA val | PANDA ISUP | PANDA+ Dice | PANDA+ ISUP | G5 prec | L_pix | L_slide | L_grade |
|---:|----------:|-----------:|------------:|------------:|--------:|------:|--------:|--------:|
| 7 | **0.608** | — | 0.587 | **62.5%** | **0.569** | 0.343 | 1.689 | 1.038 |
| 15 | 0.579 | — | 0.579 | 52.1% | 0.496 | 0.348 | 1.426 | 1.047 |
| 28 | 0.569 | 40.3% | 0.565 | 54.2% | 0.475 | 0.337 | 1.376 | 0.989 |
| **29** | 0.550 | **47.0%** | **0.609** | **60.4%** | **0.559** | 0.351 | 1.454 | 0.925 |
| 30 | 0.543 | 38.8% | 0.600 | 60.4% | 0.544 | 0.330 | 1.440 | 0.864 |
| 31 | 0.500 | 36.4% | 0.538 | 47.9% | 0.351 | 0.358 | 1.453 | 0.940 |
| 32 | 0.507 | 38.8% | 0.582 | 60.4% | 0.453 | 0.339 | 1.466 | 0.809 |
| 33 | 0.540 | 36.4% | 0.584 | 50.0% | 0.481 | 0.351 | 1.559 | 0.817 |
| 34 | 0.520 | 41.3% | 0.573 | 60.4% | 0.486 | 0.328 | 1.465 | 0.804 |

Ep8–14 and 16–27: **PANDA+ / ISUP not scored** (watcher did not backfill). Val Dice only:

| Ep | PANDA val | Ep | PANDA val | Ep | PANDA val |
|---:|----------:|---:|----------:|---:|----------:|
| 1 | 0.529 | 10 | 0.560 | 19 | 0.545 |
| 2 | 0.505 | 11 | 0.504 | 20 | 0.497 |
| 3 | 0.521 | 12 | **0.457** | 21 | 0.544 |
| 4 | 0.600 | 13 | 0.557 | 22 | 0.541 |
| 5 | 0.546 | 14 | 0.537 | 23 | 0.489 |
| 6 | 0.600 | 16 | 0.521 | 24 | 0.550 |
| 8 | 0.553 | 17 | 0.507 | 25 | 0.514 |
| 9 | 0.541 | 18 | **0.460** | 26 | 0.542 |
|  |  |  |  | 27 | 0.546 |

Val dips at ep12 / 18 / 20 / 23. That is **not** proof those epochs were bad on PANDA+.

---

## 2. Teacher A (`uni2_upernet_raw_h200x4`)

Pixel CE + soft Dice, adj-soft α=0.15, backbone freeze 3 epochs then unfrozen. **Old split** (pre–grouped fusion). No Opt3 \(L_\mathrm{slide}\).

Sweep = PANDA+ **Dice only** on every ckpt still on disk (every-5 + named 35/41/42). ep50/55/60 evals are **broken** (Dice 0.000; ignore).

| Ep | PANDA val | PANDA+ Dice | PANDA+ ISUP | PANDA ISUP |
|---:|----------:|------------:|------------:|-----------:|
| **5** | 0.602 | **0.563** | — | — |
| 10 | 0.628 | 0.526 | — | — |
| 15 | 0.639 | 0.556 | — | — |
| 20 | 0.672 | 0.536 | — | — |
| 25 | 0.668 | 0.558 | — | — |
| 30 | 0.677 | 0.560 | — | — |
| 35 | 0.726 | 0.554 | — | — |
| 40 | 0.722 | 0.541 | — | — |
| 41 | 0.730 | 0.541 | — | — |
| **42** | **0.742** | 0.554 | 31.2% *(invalid: all derived 4+3)* | **53.9%** |
| 45 | 0.669 | 0.538 | — | — |

Val climbs 0.60 → 0.74 while PANDA+ stays ~0.54–0.56. High PANDA val here is **not** external skill (split leakage + mask overfit). **No Teacher A epoch beat live ep29 on PANDA+.**

---

## 3. First Opt3 (`pseudo_r1_opt3_slidebag`)

Same Opt3 losses, **pre-Omar-6** split / wiring. Only ep12 / 15 / 21 scored on PANDA+.

| Ep | PANDA val | PANDA+ Dice | PANDA+ ISUP | Train-set ISUP |
|---:|----------:|------------:|------------:|---------------:|
| **12** | 0.619 | **0.551** | — | 52.6% |
| 15 | 0.498 | 0.451 | — | — |
| **21** | **0.636** | 0.546 | 58.3%† | **57.3%** |

Val-best (ep21) is worse on PANDA+ than ep12. Ep12 ckpt later pruned.

---

## 4. Locked Omar-6 r2 (`opt3_omar6_locked`) — early

Job **5445276**, current wiring (`WIRING_OK live=64`, LoRA in AdamW). λ_slide still warming in this window. **Not** comparable to live ep29 yet.

| Ep | PANDA val | PANDA ISUP | PANDA+ Dice | PANDA+ ISUP | G5 prec |
|---:|----------:|-----------:|------------:|------------:|--------:|
| 1 | 0.463 | 41.1% | 0.455 | 60.4% | 0.390 |
| 2 | 0.440 | 43.4% | 0.414 | 66.7% | 0.660 |
| 3 | 0.430 | 39.6% | 0.410 | 45.8% | 0.238 |
| 4 | 0.430 | 40.3% | 0.432 | 43.8% | 0.253 |
| 5 | 0.539 | 43.4% | 0.490 | 41.7% | 0.564 |
| 6 | 0.397 | 24.8% | 0.454 | 39.6% | 0.309 |

---

## 5. Other prior runs (PANDA+ Dice only unless noted)

| Run | Ep | PANDA val | PANDA+ Dice | ISUP |
|-----|---:|----------:|------------:|------|
| Rules R4 `h200x4_40k_g45soft_bf16` | 6 | 0.645 | 0.554 | — |
| Rules R4 | 15 | 0.707 | 0.540 | — |
| Rules R4 | 35 | 0.722 | 0.539 | — |
| Rules R4 | 61 | 0.739 | 0.528 | — |
| from742 `h200x4_40k_bf16_from742` | 29 | 0.752 | 0.537 | — |
| 10k g45soft | 72 | 0.716 | 0.552 | — |
| wmfix `pseudo_r1_isup_wmfix` best | 27 | 0.650 | 0.508 | — |

---

## Caveats

1. Live ep8–27 (except 15) have **no** PANDA+ / ISUP.
2. Teacher A PANDA+ ISUP at ep42 is **not usable** (collapse to 4+3).
3. Teacher A / first Opt3 / R4 used **older splits**; Omar-6 live/r2 use **grouped fusion** (0 confirmed twin leaks). PANDA val is not apples-to-apples across those eras.
4. Live LoRA was not in AdamW; locked r2 / λ015 are the wired recipe.
5. Teacher A ep50/55/60 periodic evals failed (Dice 0); omitted from leaderboard.

**Sources:** `training_log.csv` per ckpt dir; `outputs/docs/opt3_this_run/epoch_external_scorecard.csv`; `outputs/evaluation/uni2_upernet_raw_panda_plus_*_labeled.csv`; `outputs/pseudo_label/epoch_eval/` and gland-ISUP summaries.
