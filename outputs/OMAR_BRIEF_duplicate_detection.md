# Duplicate slides — brief for Dr. Omar  
**2026-08-10 · Anh · PANDA Radboud clean set (4,683 slides)**

## Why this matters
Same biopsy cores appear under different `image_id`s. Before dedupe, **~41% of val slides** had a twin sitting in train → reported metrics were inflated (val cancer Dice ~0.70 likely closer to ~0.60 when clean).

---

## What we tried

| Method | Result |
|---|---|
| **Shape (silhouette IoU)** | Failed. Needle cores all look alike. True twins found at IoU 0.30; no threshold separates twins from lookalikes. |
| **Cheap fingerprints** (phash, colour hist) | AUC looked OK (~0.92) but at usable recall **4,664 / 4,683 slides merged into one blob**. |
| **UNI2 embeddings** (24 patches/slide, full 4,683, job 5394252) | Works via **rank**, not absolute cosine. Absolute cosine collapses at full scale (median top-1 = 0.977 for everyone). |

---

## UNI2: rank-1 works; rank-2/3 collapse

- **Rank-1** (each slide’s nearest neighbour): usable. Builds **1,474 groups**.
- **Rank-2 / mutual-top-K**: chain-collapses (largest group → ~4,467). **Do not auto-add.**
- On **847 human-confirmed twin pairs**, partner’s UNI2 rank:

| Best rank | Pairs | Share |
|---|---|---|
| 1 | 676 | 80% |
| 2–5 | 82 | 10% |
| >5 | 86 | 10% |

---

## Honest scoreboard (three different numbers)

### 1. Resolved for sure (human-confirmed)
**847 twin pairs** in the ledger (Fri–Sun gallery review + a few UNI2 confirms).  
**122** adjudicated **not**-twin pairs.

### 2. Model rank-1 mutual nearest neighbours
| | Count |
|---|---|
| Mutual-NN pairs total | 1,605 |
| Already confirmed twins | 575 |
| Already confirmed not-twins | 1 (+ more FPs from ongoing review) |
| **Never human-reviewed** | **~1,029** |

Ongoing review of the weak end (cos ~0.94–0.95): moderate FPs (e.g. 8 marked on pages 102–103). Strong end: mostly true twins.  
**FP estimate on unreviewed mutual pile: ~1–5% (~10–50 pairs).**  
Large mixed-grade groups (size 5+) are mostly chain-collapse → **all pinned to train** (harmless for eval).

### 3. Residual never caught by rank-1
~20% of *known* twins are not rank-1 — but those are **already in the ledger**.  
Real hole: unknown twins never human-flagged **and** not rank-1 → roughly **~200–250 pairs**.

**Bottom line**
- Sure twins: **847**
- Likely twins in proposal, not fully reviewed: **~1,000** (≈10–50 FPs)
- Still invisible: **~200–250**
- Live `outputs/splits/` **not yet applied** — proposal only

---

## Proposed fix (replace dropping)

**Old:** delete one of each twin → lost 852 slides; pairs kept resurrecting.  
**New:** keep all **4,683** slides; put related slides in a **group**; assign whole groups to train/val/test.

- Edges = ledger twins ∪ UNI2 **rank-1** − adjudicated not-twins  
- Groups larger than 2 → **train only** (avoids one specimen dominating val/test)  
- Proposal: train **3,737** / val **472** / test **474** · every ISUP grade within **0.3%** of 80/10/10 · **0** confirmed twins split across splits  

**Eval note:** test still holds duplicate *pairs inside* it (~107 confirmed + ~130 unreviewed mutual). Score **per group** (average within group, then across groups) so each specimen votes once.

---

## Paths
- Proposal split: `outputs/docs/slide_groups/grouped_split_rank1.csv`  
- Rank-1 review gallery: `outputs/docs/slide_groups/rank1_unreviewed/`  
- Day log: `/common/omarmlab/members/anh/panda_project/DAILY_PROGRESS.md`  
- Tech log: `outputs/EXPERIMENT_LOG.md`  

**Status:** awaiting sign-off before writing live `panda_{train,val,test}.csv`.
