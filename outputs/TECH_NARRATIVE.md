# Technical narrative — PANDA Gleason pipeline

Digestible tech write-up of **what we did, why, thresholds, and results**.  
For slide bullets see `PRESENTATION_OUTLINE.md`. For day-by-day see `DAILY_PROGRESS.md`.

---

## 0. Goal in one paragraph

Build a **6-class pixel segmentation** model (background, stroma, benign, G3, G4, G5) on Radboud PANDA WSIs, then check it on held-out **PANDA+**. Path: **QC → patches → train → external eval → ISUP diagnostic → label fix → (planned) self-correct with audited re-checks**.

Final training recipe so far: **raw patches + UNI2-h + UPerNet** (Model C). Stain-norm was built at scale but **not** used for Model C training.

---

## 1. Five QC checks on whole slides

**Code:** `src/clean_dataset.py`  
**Input:** ~5060 Radboud slides with masks  
**Output:** `data/radboud_clean.csv` (after later PANDA+ exclusion: **4683**)

A slide **passes** only if all of these pass:

| # | Check | Rule | Failures (of 5060) |
|---|--------|------|---------------------|
| 1 | **Grade consistency** | Every Gleason grade named in metadata must appear as **>0 pixels** in the mask | **322** |
| 2 | **Mask completeness** | Labeled tissue ≥ **1%** of mask (`TISSUE_RATIO_MIN=0.01`) | **1** |
| 3 | **File integrity** | Mask opens with OpenSlide | **7** |
| 4 | **Benign consistency** | Metadata `negative` → **no** cancer pixels (class ≥3) | **0** |
| 5 | **Shape match** | Mask L0 size matches WSI (when slide is local) | **0** |

Also dropped if mask missing locally (**5**).  
**Total removed by QC:** **329** → **4731** pass.

**Artifacts:** `outputs/clean_dataset_flags.csv`, `outputs/clean_dataset_excluded_by_reason.csv`

**Later (not one of the 5):** exclude **48** slides that overlap PANDA+ geojson export (`data/panda_plus_ids.txt`) so external eval does not leak → **4683** clean.

**What this means:** We only train on slides where the mask and clinical grade story are at least internally consistent and the file is usable.

---

## 2. Pen-mark detection (flag-only)

### Why we needed it
Ink / pen on slides can look like tissue or confuse models. We wanted a slide-level **review queue**, not automatic deletion (pen ≠ always “bad for training”).

### Attempt 1 — HSV heuristics
- **Code:** `src/scan_pen_marks.py`
- Color thresholds for black / blue / green on a thumbnail
- Flag if pen fraction of thumb ≥ **0.0001**
- Flagged **992 / 4731 (~21%)**
- **Problem:** confirmed **false positives** on brown/olive tissue edges (not real pen strokes)

### Attempt 2 — WSISegQC `pen.pt` (what we use)
- **Code:** `src/pen_wsisegqc.py`, `src/scan_pen_marks_wsisegqc.py`
- Model: **UNet++ / ResNet34** pen segmenter (`external/wsisegqc/.../pen.pt`)
- Flag if `pen_pct_of_tissue ≥ 0.05%` **and** `pen_area_px ≥ 10`
- Flagged **671 / 4731 (~14.2%)**
- Cleared known HSV FPs; on a 20-slide sample many HSV-only flags were not real pen

**Policy:** **flag only** — does **not** remove slides from `radboud_clean.csv`.  
**Artifacts:** `outputs/pen_mark_detection_v2/` (CSV, id list, HSV vs pen.pt comparison, summary)

---

## 3. Patch extraction & keep/drop thresholds

WSIs are huge → train on **512²** patches from tissue regions (Trident / production extract path).

Important **patch-level** gates used in filtering (not the 5 slide QC checks):

| Gate | Typical value | Role |
|------|---------------|------|
| RGB tissue keep | ≥ **40%** tissue-like pixels | Drop empty / mostly glass |
| Pen in patch | black/green **>2%**, blue **>5%** | Drop heavily inked tiles |
| White / blank | high RGB / low channel Δ | Drop blank tiles |
| Cancer “rescue” | e.g. mask tissue ≥15% or cancer ≥1%, etc. | Keep rare cancer tiles that fail RGB tissue |
| Later tighten | mask tissue &lt;2% & bg ≥95% & no cancer → drop | Extra junk cull |

After splits (and PANDA+ exclusion): **~440K train** / ~56K val / ~54K test patches (order of magnitude; exact in split CSVs).

---

## 4. Stain normalization — built, not used for Model C

| | |
|--|--|
| **Method** | **Vahadane** (Macenko also supported) — `src/stain_normalize.py` |
| **Scale** | Yes — tissue-only Vahadane H5s under `outputs/stain_norm_kept/vahadane/` (~4731 slides; ~556K kept patches in one index summary) |
| **Skip if** | tissue fraction on tile &lt; **0.50** (don’t norm empty junk) |
| **Used for Model C?** | **No** — final Model C trains on **raw** RGB + ImageNet normalize only (`*_raw_*` checkpoints) |

**Why not (so far):** pipeline complexity + ink/artifact paths looked risky; Model C on **raw** already beat the early baseline on PANDA+. Stain-norm remains an option for a future Model-B-style ablation, not current training.

---

## 5. Model A (EfficientNet) → why we moved on

| | |
|--|--|
| **Architecture** | **EfficientNet-B4 + UNet++ (scse)** — Model A |
| **Example ckpt** | `outputs/checkpoints/baseline_raw_h200x3/epoch_050_dice_0.6358.pth` |
| **Internal** | Val mean Dice ~**0.64**, cancer ~**0.70** — looks fine |
| **External PANDA+** | cancer Dice ~**0.46**, weak G5 recall ~**0.35** |

**Meaning:** Strong on in-distribution val/test, **weak on external** PANDA+. That gap is why we switched to a histopathology foundation encoder.

---

## 6. Model C (current) — architecture, loss, soft labels

| | |
|--|--|
| **Architecture** | **UNI2-h** (encoder) + **UPerNet** (decoder) |
| **Input** | Raw 512², ImageNet mean/std |
| **Loss** | **0.5** weighted CE + **0.5** soft Dice (`src/train/losses.py`) |
| **Adjacent soft α = 0.15** | Keep **0.85** on true Gleason class; spread **0.15** to neighbors (G4 → 0.85 G4 + 0.075 G3 + 0.075 G5; G3 → 0.85 G3 + 0.15 G4) |
| **Optional G4↔G5 soft α = 0.22** | Stronger G4/G5 confusion allowance (ablations B / R4) |

### Ablation snapshot (same arch + same base loss)

| ID | Patches/ep | Soft | Val cancer | PANDA+ cancer | Takeaway |
|----|------------|------|------------|---------------|----------|
| **A** (teacher) | 40k | adj 0.15 | **0.742** | **0.554** | Best external so far |
| **B** | 10k | adj 0.15 + g45 0.22 | 0.716 | 0.552 | Faster; val hurts more than PANDA+ |
| **C** | 40k continue | adj 0.15 | **0.752** | 0.537 | Val↑ can be overfit |
| **R4** | 40k | adj 0.15 + g45 0.22 | ~0.707 mid-run | ep15 → 0.540 | Fair test of g45 without 10k confound |

**Meaning:** Always trust **PANDA+** when val and external disagree.

---

## 7. ISUP grade check — how we compute it from the model

**Code:** `src/isup_diagnostic.py` — `derive_grade()`, `gleason_to_isup()`  
**Teacher used:** Model C A — `epoch_042_cancer_0.7420.pth`  
**Scope:** **3746** train slides / **440371** patches  
**Report:** `outputs/pseudo_label/diagnostic_report.csv` (+ `_summary.json`)

### Algorithm (plain language)

1. Run the segmentation model on all patches of a slide; count predicted **G3 / G4 / G5** pixels.
2. Convert counts to **fractions of total cancer** (G3+G4+G5).
3. **Keep** a grade only if its fraction ≥ **`min_area_pct`** (default **5%** = `0.05`). Tiny blobs below the gate are ignored so noise doesn’t invent a secondary grade.
4. Sort remaining grades by area:
   - **Primary** = largest
   - **Secondary** = second largest, or same as primary if only one grade remains
5. Map Gleason pair → **ISUP** (standard table), e.g. 3+3→1, 3+4→2, 4+3→3, 4+4/3+5/5+3→4, 4+5/5+4/5+5→5; no cancer → 0.
6. Compare **predicted ISUP** to **metadata ISUP**.

### Results @ 5%

| | |
|--|--|
| **Overall match** | **2018 / 3746 = 53.9%** |
| ISUP 0 | ~**18%** match |
| ISUP 1 | ~**77%** |
| ISUP 2 | ~**52%** |
| ISUP 3 | ~**47%** |
| ISUP 4 | ~**67%** |
| ISUP 5 | ~**71%** |

### What it means

- About **half** of train slides disagree between **pixel-derived grade** and **slide label** — so masks are a noisy teacher for Gleason pattern, not gospel.
- **ISUP 0:** most “mismatches” are **tiny false cancer** (623 slides; median cancer/tissue ~**0.007%**, max **3.1%**, **all &lt;5%**). Visuals look like benign/stroma/ink speckles — **formula noise**, not big false tumors → **do not “correct” masks** with a tissue gate (would hurt real ISUP-1).
- **ISUP 2↔3:** often pure **3+4 ↔ 4+3** swaps → fixable with **soft** G3/G4 rebalance (**Rule B**).
- **ISUP 1 / 4 / 5 mismatches:** often locked on one wrong grade → **hard** remap toward metadata primary (**Rule C**).
- **Matches:** leave alone.

---

## 8. Changing the gate 5% → 3% (and why we reverted)

| | |
|--|--|
| **Why try 3%** | Rescue weak secondary G3 on true 4+3 cases where G3 is often **2–4%** of cancer |
| **Artifact** | `diagnostic_report_v2.csv` (`min_area_pct=0.03`) |
| **Overall match** | Still **~53.9%** — **no net gain** |
| **Trade** | **86** slides fixed, **86** broken |
| **Helps** | ISUP 2 / 3 / 5 somewhat |
| **Hurts** | ISUP 1 / 4 |
| **New failure mode** | New breaks dominated by **weak secondaries in [3%, 5%)** — same kind of formula artifacts we tried to avoid |

**Decision:** keep **`min_area_pct = 0.05`**. Lesson for later self-correct: **small threshold tweaks can rearrange errors by class without improving the headline metric** — need alerts, not silent changes.

---

## 9. Label correction plan (Phase 3) + planned train-time ISUP loop

### Offline correction (current plan)
- Source of truth: **5%** `diagnostic_report.csv` (not v2)
- **ISUP 0:** skip
- **ISUP 2↔3 validated swaps:** Rule B soft renormalize
- **Other mismatches (1/4/5, etc.):** Rule C hard → metadata primary
- **Matches:** unchanged  
- Out: `outputs/pseudo_label/corrected_labels/`, `correction_manifest.csv`

### Planned: ISUP consistency inside training (self-correct)
**Idea:** use ISUP check to update labels / loss so the model can fix itself over time.

**Risk:** if rules change continuously, you can’t tell *when* match rate or mismatch patterns shifted (same class of bug as silent 5%→3%).

**Audit protocol (planned):**
1. Every **10 epochs**, run diagnostic + correction report; save dated artifacts
2. **Pause / print summary**
3. **Alert** if match rate drops a lot vs last checkpoint, **or** a new mismatch pattern appears (e.g. [3%,5%) secondary surge)
4. Human continue / roll back → resume

---

## 10. One-page timeline

```
5060 slides
  → 5 QC checks (−329) → 4731
  → pen.pt flag 671 (review only)
  → exclude 48 PANDA+ leaks → 4683 clean
  → extract / filter 512² patches → ~440K train
  → Vahadane at scale (available) but Model C trains RAW
  → Model A EfficientNet: good val, PANDA+ cancer ~0.46 → drop
  → Model C UNI2+UPerNet + soft 0.85/0.15 → teacher A PANDA+ 0.554
  → ISUP diagnostic @5%: 53.9% match; per-grade story → Rule B/C plan
  → 3% trial: no net gain, class-shifting damage → revert 5%
  → next: Phase 3 offline fix → retrain → 10-ep ISUP audit in train
```

---

## Key paths (cheat sheet)

| Topic | Path |
|-------|------|
| QC flags | `outputs/clean_dataset_flags.csv` |
| Clean list | `data/radboud_clean.csv` |
| Pen v2 | `outputs/pen_mark_detection_v2/` |
| Stain H5s | `outputs/stain_norm_kept/vahadane/` |
| Teacher A | `outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth` |
| ISUP report | `outputs/pseudo_label/diagnostic_report.csv` |
| Soft labels | `src/train/losses.py` → `gleason_adjacent_soft_targets` |
| ISUP math | `src/isup_diagnostic.py` |
