# AGGC — what is what (for Omar)

Two models, always:
- **ZS** = PANDA Round-3 **epoch 6** (`0.3488`) — never trained on AGGC/SICAP
- **SICAP** = that model **fine-tuned** on SICAP (`0.6893`)

Two ways to score the **same** Test slides:
- **POOLED** = dump all pixels into one pile → one number (what you already sent him)
- **PER-SLIDE / PER-CORE MEDIAN** = one score per slide, then median (IQR) — Omar’s ask; **not sent yet**

Train = **extra**, never mix into official Test.

---

## A. ALREADY SHOWED HIM (Test only)

### A1. Pooled Test (the table in your screenshot)
| | wF1 | G3 | G4 | G5 | Normal |
|--|--:|--:|--:|--:|--:|
| ZS S1 (45) | 0.609 | 0.728 | 0.690 | 0.386 | 0.516 |
| SICAP S1 | 0.621 | 0.737 | 0.714 | 0.383 | 0.565 |
| ZS S2 (16) | 0.533 | 0.773 | 0.637 | 0.083 | 0.542 |
| SICAP S2 | 0.479 | 0.680 | 0.541 | 0.086 | 0.513 |
| ZS S3 (67) | 0.539 | 0.724 | 0.753 | 0.035 | 0.489 |
| SICAP S3 | 0.551 | 0.740 | 0.771 | 0.033 | 0.547 |

Plus: bootstrap, G3→G4 on S2, scanner×specimen plots, G5 slide AUC.

---

## B. SHOW HIM NEXT — Test PER-SLIDE median (same Test slides, different summary)

Label this clearly: **"per-slide median (IQR), not pooled"**

| Subset | Model | n | Median wF1 | IQR |
|--------|-------|--:|----------:|-----|
| S1 Test | ZS | 45 | 0.609 | 0.488–0.728 |
| S1 Test | SICAP | 45 | 0.623 | 0.491–0.734 |
| S2 Test | ZS | 16 | **0.664** | 0.514–0.813 |
| S2 Test | SICAP | 16 | **0.646** | 0.469–0.764 |
| S3 Test | ZS | 67 | 0.584 | 0.439–0.640 |
| S3 Test | SICAP | 67 | 0.573 | 0.465–0.709 |

One sentence for him: *S2 pooled made SICAP look bad (0.479); per-core median is ~0.65 for both — big cores were dragging the pile.*

---

## C. SATURDAY TRAIN — separate pile (extra check only, NOT official)

Do **not** pool with Test. Native raw. Per-slide CSVs.

### C1. S2 Train — per-core (37) — main reason we ran Train

| Model | n | Median wF1 | IQR | Pooled wF1 | Pooled G3 | G4 | G5 | Normal |
|-------|--:|----------:|-----|----------:|----------:|---:|---:|-------:|
| ZS | 37 | **0.628** | 0.515–0.727 | 0.555 | 0.799 | 0.650 | 0.117 | 0.557 |
| SICAP | 37 | **0.551** | 0.466–0.646 | 0.513 | 0.717 | 0.590 | 0.113 | 0.530 |

Same pattern as S2 Test: **ZS ahead of SICAP** on biopsies.

### C2. S1 Train — extra whole mounts (optional)

| Model | n | Median wF1 | IQR | Pooled wF1 | Pooled G3 | G4 | G5 | Normal |
|-------|--:|----------:|-----|----------:|----------:|---:|---:|-------:|
| ZS | 94 | 0.623 | 0.473–0.773 | 0.556 | 0.706 | 0.656 | 0.283 | 0.505 |
| SICAP | 94 | 0.610 | 0.476–0.742 | 0.549 | 0.705 | 0.664 | 0.243 | 0.528 |

Close; ZS slightly higher median. (94/105 images on disk.)

### C3. S3 Train — more specimens × scanners

| Model | n | Median wF1 | IQR | Pooled wF1 | Pooled G3 | G4 | G5 | Normal |
|-------|--:|----------:|-----|----------:|----------:|---:|---:|-------:|
| SICAP | 144 | 0.626 | 0.525–0.664 | **0.627** | 0.801 | 0.791 | 0.251 | 0.553 |
| ZS | 144 | — | — | **rerunning** job **6476097** | — | — | — | — |

SICAP done. ZS OOM’d Saturday; rerun in progress (~14 min in).

---

## Cheat: which number to quote

| Question | Use |
|----------|-----|
| Official / challenge-style | **Test pooled** (section A) |
| Omar median / per-core | **Test median** (section B) |
| “Did Train look the same?” | **Train** (section C) — optional |

---

## PAPER DECISION (Omar, 2026-11-11) — LOCKED

> “with everything at native raw it seems fine tuning on SICAP neither helped nor hurt AGGC significantly. lets keep the zero shot model as the primary model in the paper since it didn’t see both datasets and SICAP fine tuning can go in supp or something”

**Primary (main paper):** PANDA Option-3 Round-3 **epoch 6 zero-shot**  
`epoch_006_cancer_0.3488.pth` — never trained on SICAP or AGGC.

**Supplement only:** SICAP-FT replay-0.3 ep6  
`epoch_006_cancer_0.6893.pth` — shows FT neither clearly helps nor hurts AGGC at native raw.

Do not reframe SICAP-FT as the AGGC headline model unless Omar revisits this.
