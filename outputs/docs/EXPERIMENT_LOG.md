# Experiment Log — PANDA Gleason Segmentation (Model C)

Living log of runs, changes, and decisions. **Update whenever a training/eval/protocol change lands.**  
Day-by-day Cursor log: `outputs/DAILY_PROGRESS.md` (resume text never goes here).

Format: **Why → What → Result → Decision**

---

## Active defaults (current)

| Item | Value |
|------|--------|
| Architecture | UNI2-h + UPerNet |
| Input | raw 512² (ImageNet norm) |
| Loss | 0.5 CE + 0.5 soft Dice; adj soft α=0.15; optional g45 α=0.22 |
| Classes | 0 bg, 1 stroma, 2 benign, 3 G3, 4 G4, 5 G5 |
| **Teacher / external report** | `…/h200x4/epoch_042_cancer_0.7420.pth` (PANDA+ cancer **0.554**) |
| PANDA+ protocol | `gt≥2` only; benign/G3/G4/G5; `--panda-plus-eval` |
| ISUP diagnostic `min_area_pct` | **0.05** |
| Phase 3 ISUP-0 | **skip** |
| Handoff README | `/common/omarmlab/members/anh/panda_project/README.md` |

---

## Checkpoint leaderboard

| ID | Stem | PANDA val cancer | PANDA+ cancer | Role |
|----|------|-----------------:|--------------:|------|
| **A** | `h200x4/epoch_042_cancer_0.7420` | **0.742** | **0.554** | Best external teacher |
| B | `10k_g45soft_fp32/epoch_072_cancer_0.7160` | 0.716 | 0.552 | Near-tie PANDA+ |
| C | `40k_bf16_from742/epoch_029_cancer_0.7521` | 0.752 | 0.537 | Val overfit |
| **R4** | `40k_g45soft_bf16/epoch_061_cancer_0.7394` | **0.739** | **0.528** | Fair 40k+g45; external below A |
| R4-ep35 | `40k_g45soft_bf16/epoch_035_cancer_0.7222` | 0.722 | **0.539** | Best R4 PANDA+ |
| R5 | `h200x4_10k_adj015` | — | — | Failed job `5260610`; resubmit after code restore |

---

## Ablation honesty

A/B/C were sequential engineering runs (confounded). Cleaner grid:

| | adj 0.15 only | adj 0.15 + g45 0.22 |
|--|--|--|
| **40k** | A-like (messy history) | **R4** |
| **10k** | **R5** (pending resubmit) | **B** |

---

## R4 — 40k + g45 soft (done)

| | |
|--|--|
| **Why** | Isolate g45 soft vs teacher A without 10k confound |
| **What** | 4×H200; 40k/ep; adj 0.15 + g45 0.22; BF16; job **5247423** |
| **Result** | Early stop ep81 (patience 20); best val **0.7394 @ ep61** |
| **PANDA+** | ep35 **0.539**; ep61 **0.528** |
| **Decision** | Do not promote over A (0.554) |

---

## R5 — 10k + adj 0.15 only (failed once)

| | |
|--|--|
| **Why** | Complete 10k × adj-only cell of the ablation grid |
| **What** | Job **5260610**; script `scripts/slurm_train_uni2_10k_adj015_h200x4.sh` |
| **Result** | Started on cp098 Jul 23 → **FAILED** (~9s) on corrupt/syntax-broken `train_uni2_upernet.py` then on disk |
| **Decision** | Code restored from git 2026-07-26; resubmit when ready |

---

## Code wipe / restore (2026-07-26)

| | |
|--|--|
| **Why** | Working tree deleted critical UNI2 train/eval sources (~5k LOC) while still in git |
| **What** | `git checkout HEAD -- src/ …`; restore Phase 3 script; add README |
| **Result** | Runnable sources back on disk |
| **Decision** | Commit immediately; prefer push so wipe cannot recur |

---

## Phase 3 / ISUP-in-loss (open with Omar)

1. Offline Rule B/C soft `.npy` from 5% diagnostic (`scripts/phase3_apply_corrections.py`)
2. Wire corrected soft targets into training dataset/loss
3. Optional: 10-ep ISUP diagnostic + alert loop (no silent mutations)
4. Re-check per-ISUP precision/recall after retrain (not match rate alone)

---

## Open next

1. Push commits to origin  
2. Resubmit R5  
3. Clean Phase 3 from 5% report  
4. Omar design session on loss-loop vs offline-only  
