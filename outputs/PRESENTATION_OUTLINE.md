# Presentation outline — PANDA Gleason segmentation

Keep each slide short: **goal → what → why → result**. Extra detail slides are marked.  
**Full tech narrative (QC → pen → stain → EfficientNet → ISUP math):** `TECH_NARRATIVE.md`

---

## 1. Title / problem (1 slide)
- **What:** Pixel-level Gleason grade segmentation on prostate WSIs
- **Why it matters:** Slide-level ISUP alone is coarse; pixel maps support grading, QC, and trust
- **Task:** 6-class segmentation — background, stroma, benign, G3, G4, G5

## 2. Scale & data (1 slide)
- **~4.7K** Radboud clean whole-slide images
- **~440K** training patches (512²)
- External check: **PANDA+** expert-annotated slides (held out)

## 3. Pipeline overview (1 slide)
```
QC → Patch extraction → Train Model C → Eval (val + PANDA+)
  → ISUP diagnostic → Label correction → (planned) ISUP-aware train + periodic re-check
```

## 4. Phase A — Data QC (1–2 slides)
| | |
|--|--|
| **Why** | Bad masks / ink / grade–mask mismatches poison training |
| **What** | Mask completeness, grade consistency, pen-mark detection (WSISegQC `pen.pt`), exclude PANDA+ leak |
| **Result** | Clean cohort for training; pen flagged for review (not auto-deleted) |

## 5. Phase B — Patch extraction (1 slide)
| | |
|--|--|
| **Why** | WSIs are too large to train end-to-end |
| **What** | Tissue-aware 512² patches; Trident vs naive grid → production extract |
| **Result** | Train/val/test splits; ~440K train patches |

## 6. Phase C — Model + loss (1–2 slides)
| | |
|--|--|
| **Architecture** | **Model C = UNI2-h + UPerNet** |
| **Input** | Raw 512² RGB (ImageNet norm); no stain-norm in final recipe |
| **Loss (shared across ablations)** | 0.5 weighted CE + 0.5 soft Dice |
| **Adjacent soft labels** | α = **0.15** → true class **0.85**, remainder split to neighbors (e.g. G4 → 0.85 G4 + 0.075 G3 + 0.075 G5; G3 → 0.85 G3 + 0.15 G4) |
| **Optional G4↔G5 soft** | α_g45 = **0.22** (stronger swap between G4 and G5 only) |

## 7. Ablation: 4 training recipes (2 slides)

Same architecture + same base loss; change **patches/epoch** and **soft-label recipe**.

| ID | Recipe | Patches/ep | Soft labels | Val cancer | PANDA+ cancer | Role |
|----|--------|------------|-------------|------------|---------------|------|
| **A** | Baseline teacher | **40k** | adj **0.15** only | **0.742** | **0.554** | Best external → Phase 2/3 teacher |
| **B** | Faster + G45 soft | **10k** | adj 0.15 + **g45 0.22** | 0.716 | 0.552 | ≈A on PANDA+; worse val/G5 |
| **C** | Continue from A (BF16) | **40k** | adj 0.15 | **0.752** | 0.537 | Val overfit; do **not** replace A |
| **R4** | Fair A/B (in progress) | **40k** | adj 0.15 + **g45 0.22** | ~0.707 so far | ep15 → 0.540 | Isolates g45 soft without 10k confound |

**Talking points**
- 10k vs 40k: faster epochs, but val drops more than PANDA+
- Soft 0.85 / 0.15 encodes grade uncertainty between neighbors
- Always rank by **PANDA+**, not val alone (C proves the trap)

## 8. Evaluation protocol (1 slide)
- Internal: PANDA val cancer / per-class Dice
- External: PANDA+ (`gt≥2`; report benign + G3–G5)
- Reporting metric: **PANDA+ cancer Dice**

## 9. ISUP grade check — what it is (1 slide)
| | |
|--|--|
| **What** | Run teacher on each **train slide** → measure G3/G4/G5 area → derive primary+secondary Gleason → map to **predicted ISUP** → compare to **metadata ISUP** |
| **Gate** | Ignore tiny grade blobs below `min_area_pct` (**5%** of tissue) so formula noise doesn’t invent a secondary grade |
| **Why** | Pixel masks often disagree with slide-level grade → noisy supervision for learning |

**Overall (teacher A, 3746 slides):** match **53.9%** (2018 / 3746)

| Metadata ISUP | Match rate | Read |
|---------------|------------|------|
| 0 | ~18% | Many tiny “cancer” FPs — almost all **&lt;5%** of tissue (noise, not real tumors) |
| 1 | ~77% | Best agreement |
| 2 | ~52% | Often confused with 3 (3+4 ↔ 4+3) |
| 3 | ~47% | Same 2↔3 swap problem |
| 4 | ~67% | Often locked on wrong primary |
| 5 | ~71% | Fairly stable |

**5% → 3% threshold trial:** net match **unchanged** (86 fixed / 86 broken); new breaks = weak secondaries in **[3%,5%)**. **Keep 5%.**

## 10. Detailed fix plan by ISUP (1–2 slides)

| Metadata ISUP | Finding | Fix plan |
|---------------|---------|----------|
| **0** | Speckle FPs only; all &lt;5% tissue | **No mask correction** (tissue gate would hurt real ISUP-1). Optional later: aux “cancer-present” head |
| **1** | High match; residual mismatches often locked wrong primary | **Rule C:** hard remap cancer pixels → metadata primary (when mismatch) |
| **2 ↔ 3** | Only pair that is cleanly **swap-fixable** (3+4 ↔ 4+3) | **Rule B:** soft renormalize G3/G4 mix to match metadata pattern (don’t hard-kill the secondary) |
| **4** | Mismatches often ~95–100% one wrong grade | **Rule C:** hard → metadata primary |
| **5** | Similar locked-primary pattern | **Rule C:** hard → metadata primary |
| **Matches** | Already consistent | **Leave labels unchanged** |

Phase 3 status: offline correction from **5%** diagnostic (restart; do not use 3% v2).

## 11. Planned next: ISUP check inside training (self-correct) (1–2 slides)

**Idea:** Feed ISUP consistency into the **loss / label stream** so the model can self-correct noisy pixels over time.

**Problem:** If labels/loss rules change continuously, we can’t tell *when* match rate or mismatch *patterns* changed (same failure mode as the silent 5%→3% threshold change).

**Protocol (planned)**
1. Every **10 epochs**: freeze a checkpoint → run full **ISUP diagnostic** + **correction report**
2. Save artifacts: `diagnostic_report_epXXX.csv`, `correction_manifest_epXXX.csv`, short summary JSON
3. **Pause & print a summary alert** if:
   - overall **match rate drops significantly** vs last checkpoint, **or**
   - a **new mismatch pattern** appears (e.g. surge of weak secondaries like the 3% trial, ISUP-0 explosion, 2↔3 collapse)
4. Human reviews the alert → continue / roll back soft-label or correction settings → resume

**Goal:** self-correction **with** auditability every 10 epochs, not a silent moving target.

## 12. What’s next (near-term) (1 slide)
1. Finish Phase 3 offline corrections (5% report) → confirm → retrain
2. Ship 10-epoch ISUP diagnostic + alert loop in train
3. Optional: aux cancer-present head for ISUP-0 FPs
4. Re-check PANDA+ after each major change

## 13. Takeaways (1 slide)
- Full pipeline: QC → patches → foundation seg → external eval → ISUP feedback
- Soft labels **0.85 / 0.15** (+ optional G45) encode grade ambiguity
- Val can lie; **PANDA+** ranks teachers
- ISUP check finds *where* labels fail; per-grade rules + **10-ep audits** keep self-correction honest

---

### Optional appendix
- HPC: 4×H200 DDP, BF16, Slurm
- Full leaderboard CSVs under `outputs/evaluation/`
- Soft-label formulas in `src/train/losses.py`
