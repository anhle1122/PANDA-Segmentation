# Option 3 — epoch results & PANDA+ summary

**Tag:** `pseudo_r1_opt3_slidebag`  
**Updated:** 2026-08-05  
**Ckpt dir:** `outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag/`  
**Train log:** `training_log.csv` (epochs 1–30)

## What this run is

| Piece | Setting |
|-------|---------|
| Model | UNI2-h + UPerNet, **GN** decoder, **LoRA** QKV (backbone frozen) |
| Targets | **Original** Radboud masks (no Rules rewrite) |
| Loss | \(L_\mathrm{pixel}\) + \(\lambda_\mathrm{slide} L_\mathrm{slide}\) + \(\lambda_\mathrm{grade} L_\mathrm{grade}\) |
| λ schedule | \(\lambda_\mathrm{slide}=0\) ep1–5 → ramp ep6–9 → **0.3** from ep10; \(\lambda_\mathrm{grade}=0.3\) always |
| Gate | soft `min(1, n/64)` on slide/grade losses; live=64 |
| Judge | Prefer **PANDA+** over val / train-ISUP (standing rule) |
| Teacher A ref | val 0.742 / PANDA+ cancer **0.554** (`epoch_042`) |

\(L_\mathrm{slide}\) = soft ISUP from **paint** (not hard `derive_grade`).  
Hard drive-ISUP: `derive_grade` default **min_area_pct=0.0** (Omar 2026-08-06; was 0.05). Opt3 hard-skips L_slide/L_grade when **n&lt;5**.

---

## Stability note — yes, training is unstable

Val **cancer Dice** swings hard after \(\lambda_\mathrm{slide}\) turns on:

| Ep | Val cancer | Δ vs prev | Likely driver |
|---:|----------:|----------:|---------------|
| 10 | 0.490 | **−0.064** | \(\lambda_\mathrm{slide}\) hits **0.3** |
| 12 | **0.619** | recovery | brief sweet spot |
| 15 | 0.498 | **−0.107** | largest collapse |
| 21 | **0.636** | peak val | later rebound |
| 23 / 25 / 30 | 0.56 / 0.55 / 0.55 | repeated dips | no clear plateau |

Range over ep1–30: **0.395 → 0.636**. Not a smooth climb — consistent with slide-loss fighting pixel CE/Dice (and soft bag gate).

**Implication:** do not trust “last epoch” or val-best alone; keep multiple ckpts (prune-keep-all is now the default).

---

## All epochs — in-domain validation

From `training_log.csv`:

| Ep | Val cancer | Val mean | train_loss | val_loss | Note |
|---:|----------:|---------:|-----------:|---------:|------|
| 1 | 0.395 | 0.572 | 0.702 | 0.300 | |
| 2 | 0.569 | 0.681 | 0.580 | 0.287 | |
| 3 | 0.579 | 0.689 | 0.528 | 0.290 | |
| 4 | 0.542 | 0.666 | 0.520 | 0.288 | |
| 5 | 0.604 | 0.707 | 0.497 | 0.277 | early peak; λ_slide still 0 |
| 6 | 0.589 | 0.698 | 0.591 | 0.278 | λ ramp starts |
| 7 | 0.557 | 0.674 | 0.625 | 0.283 | |
| 8 | 0.583 | 0.695 | 0.693 | 0.276 | |
| 9 | 0.553 | 0.672 | 0.697 | 0.285 | |
| 10 | 0.490 | 0.634 | 0.746 | 0.309 | λ_slide → 0.3; **dip** |
| 11 | 0.563 | 0.680 | 0.763 | 0.282 | |
| **12** | **0.619** | **0.717** | 0.823 | 0.275 | best PANDA+ (ckpt **pruned**) |
| 13 | 0.582 | 0.698 | 0.762 | 0.279 | |
| 14 | 0.604 | 0.712 | 0.715 | 0.277 | |
| 15 | 0.498 | 0.643 | 0.733 | 0.282 | **big dip** |
| 16 | 0.578 | 0.685 | 0.717 | 0.277 | |
| 17 | 0.575 | 0.690 | 0.828 | 0.275 | |
| 18 | 0.553 | 0.681 | 0.729 | 0.279 | |
| 19 | 0.568 | 0.686 | 0.774 | 0.273 | |
| 20 | 0.598 | 0.704 | 0.772 | 0.277 | |
| **21** | **0.636** | **0.728** | 0.712 | 0.269 | val-best; on disk |
| 22 | 0.610 | 0.715 | 0.725 | 0.267 | |
| 23 | 0.562 | 0.681 | 0.651 | 0.277 | |
| 24 | 0.606 | 0.712 | 0.667 | 0.272 | |
| 25 | 0.548 | 0.677 | 0.666 | 0.275 | on disk |
| 26 | 0.564 | 0.686 | 0.719 | 0.269 | |
| 27 | 0.619 | 0.721 | 0.773 | 0.270 | |
| 28 | 0.584 | 0.697 | 0.665 | 0.271 | |
| 29 | 0.602 | 0.707 | 0.692 | 0.270 | |
| 30 | 0.547 | 0.672 | 0.647 | 0.274 | on disk (`latest`) |

**Still on disk:** `epoch_021…`, `025`, `030`, `best.pth`, `latest.pth`.  
**Pruned (lost):** `epoch_012…0.6190.pth` (and other intermediate `epoch_*.pth`).

---

## PANDA+ — only epochs we actually evaluated (Opt3)

Protocol: labeled pixels only (`gt≥2` classes benign/G3/G4/G5).  
Artifacts under `outputs/evaluation/uni2_upernet_raw_panda_plus_epoch_*_labeled*.csv`.

### Headline

| Ep | Val cancer | PANDA+ cancer | PANDA+ mean | Cancer recall | G5 recall | Train ISUP match |
|---:|----------:|--------------:|------------:|--------------:|----------:|-----------------:|
| **12** | 0.619 | **0.551** | 0.580 | 0.467 | 0.474 | **52.6%** (1970/3746) |
| 15 | 0.498 | 0.451 | 0.503 | 0.387 | 0.611 | — |
| **21** | **0.636** | 0.546 | 0.573 | 0.461 | 0.467 | **57.3%** (2145/3746) |
| Teacher A | 0.742 | **0.554** | — | — | — | 53.9% |

**Model-C pattern:** ep21 better val + train-ISUP, **worse PANDA+** than ep12.  
→ For external claims prefer **ep12 numbers**; val-best ≠ external-best.

### Per-class precision / recall / Dice (PANDA+)

**Ep12** (best Opt3 external):

| Class | Precision | Recall | Dice |
|-------|----------:|-------:|-----:|
| Benign | 0.801 | 0.572 | 0.667 |
| G3 | 0.793 | 0.500 | 0.613 |
| G4 | 0.683 | 0.433 | 0.530 |
| G5 | 0.554 | 0.474 | 0.511 |

**Ep15** (collapse epoch):

| Class | Precision | Recall | Dice |
|-------|----------:|-------:|-----:|
| Benign | 0.821 | 0.548 | 0.657 |
| G3 | 0.865 | 0.355 | 0.503 |
| G4 | 0.530 | 0.370 | 0.436 |
| G5 | 0.314 | 0.611 | 0.415 |

**Ep21** (val-best):

| Class | Precision | Recall | Dice |
|-------|----------:|-------:|-----:|
| Benign | 0.774 | 0.569 | 0.656 |
| G3 | 0.829 | 0.445 | 0.579 |
| G4 | 0.639 | 0.475 | 0.545 |
| G5 | 0.571 | 0.467 | 0.513 |

### Not Opt3

Other `panda_plus_epoch_*` files (ep006@0.645, ep029, ep035, ep042, ep061, ep072, …) are **different runs** (teacher A / R4 / etc.). Do not mix into this table.

### PANDA+ ISUP (slide-level)

First attempt on 48 matched PANDA+ slides vs `train.csv` clinical: Teacher A collapsed (all derived 4+3). Opt3 ep21 job was resubmitted; **do not cite a PANDA+ ISUP match rate** until verified. In-domain ISUP above is train-set only.

---

## How to read this for drive-ISUP / next steps

1. **Seg is jumpy under dual-ISUP** — especially when \(\lambda_\mathrm{slide}\) reaches 0.3 (ep10+) and around ep15.  
2. **Train ISUP ↑ does not imply PANDA+ ↑** (ep12 vs ep21).  
3. Omar override: **min_area_pct=0.0**; tiny-bag skip **n&lt;5** for dual ISUP.  
4. Next levers (from Part2 + this curve): hard skip \(n<32\), restrain/select by PANDA+, freeze ckpts, optional Round-2 cleaned targets for confident mismatch — not conf-gating alone.

---

## File index

| What | Path |
|------|------|
| Train log | `outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag/training_log.csv` |
| PANDA+ ep12/15/21 | `outputs/evaluation/uni2_upernet_raw_panda_plus_epoch_{012,015,021}_*_labeled.csv` |
| ISUP train ep12/21 | `outputs/pseudo_label/diagnostic_report_opt3_ep{12,21}.csv` (+ `_summary.json`) |
| Status snapshot (older) | `outputs/OPT3_STATUS_SNAPSHOT.md` |
| Experiment log | `outputs/docs/EXPERIMENT_LOG.md` |
