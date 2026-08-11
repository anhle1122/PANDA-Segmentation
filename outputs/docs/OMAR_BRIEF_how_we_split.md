# How we split the data — note for Dr. Omar  
**2026-08-11 · Anh · PANDA Radboud clean set (4,683 slides)**

## Original split (what we used before)

**Package:** `sklearn.model_selection.train_test_split`  
**Script:** `src/make_train_splits.py`

**Method:**
- Split by **slide** (not by patch)
- Target **80% / 10% / 10%** train / val / test
- Stratified on **`isup_grade`** so each split keeps the same grade mix
- Two-step call: first 80% vs 20%, then split the 20% in half → 10% val / 10% test
- Random seed fixed (`random_state=42`)

**What we did *not* use:** `GroupKFold` / `StratifiedGroupKFold`. Groups were not defined, so duplicate slides (same biopsy under different `image_id`s) could land on opposite sides of the split.

**Why that mattered:** audit found **~41% of validation slides** had a near-duplicate sitting in train → validation metrics were inflated.

**After the old “drop one twin” pass:** live files still have **3,831** slides (4,683 − 852 drops). That drop policy is what we are replacing.

---

## New split policy (applied to live `panda_*.csv` on 2026-08-11)

**Goal:** keep **all 4,683** slides; **never drop** duplicates; prevent train↔val/test leakage by moving related slides together.

### Step 1 — Build groups (similar slides stay together)

A slide group is a connected component linked by any of:

1. **Human-confirmed twin pairs** (ledger from gallery review)  
2. **UNI2 embedding mutual nearest neighbours** (each slide’s top match is the other)  
3. **UNI2 rank 2–5 + shape IoU ≥ 0.29** (same ISUP) — catches twins that embedding alone ranks just below #1, gated by silhouette overlap so we don’t chain the whole dataset  

Adjudicated “not twin” pairs are excluded as edges.

**No sklearn grouping class here** — custom union-find over those edges (`src/build_fusion_groups.py`). Embeddings: UNI2-h, 24 patches/slide. Shape: pairwise silhouette IoU (9 orientations).

### Step 2 — Assign groups to train / val / test

Still target **80 / 10 / 10**, still **stratified by ISUP grade**.

**Hybrid fill (preferred):**
1. Put **singletons** (no detected partner) into val/test first  
2. If a grade is short of 10%, add a few **size-2** groups into val/test  
3. Everything else (larger groups, remaining pairs) → **train**

So val/test are mostly “clean” slides, with a limited number of twin-pairs allowed only to hit the quota. Large similarity-chain blobs stay in train (false merges only hurt by stuffing train, not by leaking).

**Leak check on known twins:** **0** of ~850 confirmed twin pairs cross splits under this assignment.

**Eval note:** when a size-2 group is in val/test, report metrics **per group** (average within the pair, then across groups) so one specimen does not count twice.

---

## Packages / tools (summary)

| Piece | What we use |
|---|---|
| Original stratified slide split | **`sklearn.model_selection.train_test_split`** |
| New grouped assignment | Custom (union-find + ISUP quotas) — same *idea* as GroupKFold, not the sklearn API yet |
| Similarity | UNI2-h embeddings + shape IoU |
| Canonical twin / not-twin decisions | Project ledger CSVs (`twin_ledger.py`) |

We can switch the assigner to **`sklearn.model_selection.StratifiedGroupKFold`** later if you want the split to go through a standard API; the group IDs we built are exactly what that expects.

---

## Status (applied 2026-08-11)

| | |
|---|---|
| Working set | **4,683** slides (5,060 with masks − QC − PANDA+ holdout) |
| Live splits on disk | **Applied** — train **3739** / val **472** / test **472** |
| Grouping | Fusion: ledger twins ∪ UNI2 mutual-NN ∪ (rank 2–5 + IoU≥0.29), `max-eval-group=2` → large groups **train only** |
| Leak check | **0** / 850 confirmed twin pairs cross splits; val/test largest group **2** |
| Backup of drop-era 3,831 | `outputs/splits/panda_*_pre_grouped_fusion_iou0.29_rank2to5_maxeval2_2026-08-11.csv` |

**Residual risk (honest):** grouping removes leakage for every pair we link. Unknown twins that never fire mutual/fusion (~order **~4%** of estimated twin pairs, after split chance) could still cross — much smaller than the old ~41% val contamination, not mathematically zero.

**Eval:** report val/test metrics **per group** (average within group, then across groups).

---

## Paths
- Day log: `/common/omarmlab/members/anh/panda_project/DAILY_PROGRESS.md`  
- Duplicate brief: `outputs/docs/slide_groups/OMAR_BRIEF_duplicate_detection.md`  
- Fusion groups: `outputs/docs/slide_groups/slide_groups_fusion_iou0.29_rank2to5.csv`  
- Original splitter: `src/make_train_splits.py`
