# Experiment Log — PANDA Gleason Segmentation (Model C)

Living log of runs, changes, and decisions. **Update whenever a training/eval/protocol change lands.**  
Day-by-day Cursor log: `outputs/DAILY_PROGRESS.md` (resume text never goes here).

Format: **Why → What → Result → Decision**

---

## Mask ISUP vs clinical — provenance fix (2026-07-30)

**Why:** Chat reused **2018/3746 = 53.9%** as “mask-derived vs clinical,” but that number is identical to teacher A’s model-vs-clinical diagnostic.

**What:** Traced `outputs/pseudo_label/diagnostic_report.csv` — columns are `pred_pixels_*` from `src/isup_diagnostic.py` (model inference), match sum **2018**. Round1 `match` count is the same comparison. Submitted CPU job **5326054**: `src/mask_isup_vs_clinical.py` aggregates **raw mask patch** pixels → `derive_grade()` @ 5% → clinical ISUP; out `outputs/pseudo_label/mask_isup_vs_clinical.csv`.

**Result:** **5326054** finished. Raw-mask → `derive_grade` @5% vs clinical: **3054 / 3746 = 81.5%**. Teacher A model-vs-clinical remains **2018 / 3746 = 53.9%**. Not the same number — prior chat reuse was wrong. Per-metadata ISUP mask match: 0=1.00, 1≈1.00, 2≈0.55, 3≈0.54, 4≈0.96, 5≈0.77. Artifact: `outputs/pseudo_label/mask_isup_vs_clinical.csv`.

**Decision:** Cite **81.5%** as mask-vs-clinical and **53.9%** as model-vs-clinical separately. Teacher A is *worse* than the masks at matching clinical ISUP — model adds disagreement beyond mask noise (esp. ISUP 2/3).


## Option 3 — slide-bag + dual ISUP (started 2026-07-30)

**Why:** Omar Option 3 — seg CE+Dice + derived-ISUP-from-paint slide loss + separate grade-head loss; no Rules 1–3. Evaluate dual-ISUP on its own vs **teacher A**, not vs `wmfix`.

**What:** Regroup existing patches (max 323/slide); micro-batch accumulate; λ_slide=λ_grade=0.3; tag `pseudo_r1_opt3_slidebag`. \(L_\mathrm{slide}\) = soft linear proxy on \((f_3,f_4,f_5)\), **not** differentiable `derive_grade()`. Backbone UNI2 pretrained + freeze 5 ep. 256 slides/ep/rank. Adjacent soft **α=0.15**. Train on **2×H200** (`--mem=120G`) to share mixed nodes; Slurm auto-resumes `latest.pth`. Monitor `scripts/monitor_opt3_h200.sh` upgrades to 4×+400G when a full node frees (resume, never scratch).

**Result:** First start **5323068** on cp098 (preemptable) crashed: BatchNorm N=1 on remainder micro-batch (`[1,512,1,1]`). Fixed by BN-safe pad (duplicate singleton → N=2; loss on real only). Resubmitted. H200 requires `--partition=preemptable --qos=part_preemptable` (gpu+normal → ReqNodeNotAvail). Later OOMs / preempts; retuned to lazy bags + **96G** + max 160 patches/slide. **5324260** ran ~51 min then cancelled: host MaxRSS climbed linearly **~1.2G/30s** to **~67G** — uncapped H5/OpenSlide caches. Full mem fix: class LRU `max_cached_opens=2` + train-bag clear + `val_ds.clear_open_handles()`. **5326090** (full fix) FAILED @ **53m** on Jul 30 19:07: NCCL ALLREDUCE timeout 600s (rank desync / SIGABRT); never wrote ep1 metrics; no `latest.pth`. Host MaxRSS stayed **~10→13G**/96G (cache fix held). ETA rough **20–40 min/epoch** on 2× once stable; refine after ep1.

**Decision:** Judge Option 3 vs teacher A PANDA+ **0.554** cancer / **0.528** G5. **Core hypothesis:** can \(L_\mathrm{slide}\) (λ=0.3, aggregate→seg grads) counteract original-mask G3/G4 bias under full CE+Dice, with no Rules? Post-run require teacher-A-style **G3→G4 leak** analysis, not Dice alone. If underperforms / leak stuck: (1) **WeGleNet LSE** pool with val-tuned \(r\) (paper \(r=8\)) + consider secondary damp \(d\) (paper \(0.70\)) — current loss is mean-softmax + linear ISUP scores, not LSE; (2) soft-`derive_grade` surrogate; (3) higher λ / Rules+dual-ISUP same-α — before rejecting the idea. **Immediate:** diagnose DDP hang (asymmetric bag sizes / collective order across ranks) and resubmit; mem path is no longer the blocker.


## Code recovery (2026-07-30)

**Why:** Pseudo-label Round 1 sources were never committed and were wiped from disk with other train files.

**What:** Restored exact files from Cursor local History (not `.pyc`); committed as `e9add14` (`pseudo_label_rules.py`, `pseudo_label_dataset.py`, `round_control.py`, ISUP-informed losses, train CLI, Slurm/smoke/cache, protocol docs, round1 manifest).

**Decision:** Do not start Option 3 until smoke tests pass on recovered code. Commit pseudo-label / Option-3 files early and often.


## R6 — Pseudo-label Round 1 (Rules 1–3)

**Why:** Replace A/B/C/D with iterative self-training; Round 1 trains from scratch with ISUP-informed single loss (Rules 1–3 rewrite flagged pixels in the original-mask seg target).

**What:** Manifest initially 629 correcting / 3746 slides (R1: 249, R2: 9, R3: 371). Cached teacher A argmax for correcting slides. After protocol fix: wide-margin 2↔3 no longer hard-corrects; **187** slides tagged `wide_margin_unresolved` (empty flags; original mask kept). Correcting set = **442** (`rule1_soft_tie` 62 + R2 9 + R3 371).

**Result (BN / loss path):**
- `5305879` (`pseudo_r1`): cancelled; peak cancer **0.411 @ ep3**; decode BN running-stats → val NaN after unfreeze.
- `5307259` (`pseudo_r1_bnfix`): dual-loss era; superseded.
- `5307347` (`pseudo_r1_isup_seg`): ISUP single-loss + BN fix; reached **ep12**, best val cancer **0.570**; still used old `rule1_wide_margin` hard G3↔G4 rewrite → **cancelled**.
- `5307779` (`pseudo_r1_isup_wmfix`): resubmitted with `wide_margin_unresolved` manifest. **Canonical Round 1 clean baseline** — do not conflate with `5307347`.

**`5307779` isolation checklist (confirmed live from job log + script):**
- `seg_target = original_mask` only (Round 1; no `--seg-target-dir`)
- Loss = ISUP-informed single loss: Rules rewrite flagged pixels in that mask target (442 slides: soft_tie 62 + R2 9 + R3 371)
- Wide-margin 187 slides: `wide_margin_unresolved`, empty flags, no rewrite
- **OEEM deliberately OFF** (helpers exist in `losses.py`; not called from train loop) — Round 1 is Rules-only

**Protocol decision (2026-07-29):** Wide-margin Rule 1 must **not** hard-correct over-extended G3/G4 pixels — no reliable pixel-level split of real secondary vs over-extension. Same effect as missing-class / `none`, but named `wide_margin_unresolved` for audit. Soft-tie (margin &lt; 0.15) unchanged.

**Decision:** Train Round 1 on the unresolved-wide-margin manifest; judge vs teacher A on fresh `--panda-plus-eval` after `5307779`. Treat `5307347` as a discarded buggy protocol run only.

**Omar protocol (2026-07-29) — supersedes EMA plan:** Round-based Option A only; **reject** mid-run self-relabel (B). Pixel mask regenerates once/round; continuous honesty = **live ISUP grade head/loss** every batch (not a second fighting pixel loss). Round 2+ target = cleaned map (agree→keep label; disagree+low conf→ignore; disagree+conf→ISUP referee then swap). See `outputs/pseudo_label/OMAR_ROUND_PROTOCOL.md`. Keep ≤2 rounds. Current `5307779` remains Round 1 mask-start baseline (grade head not yet built).



---

## Active defaults (current)

| Item | Value |
|------|--------|
| Architecture | UNI2-h + UPerNet |
| Input | raw 512² (ImageNet norm) |
| Loss (baseline) | 0.5 CE + 0.5 soft Dice; adj soft α=0.15–0.22 |
| Loss (pseudo Round 1+) | `combined_loss` w_seg=0.70 / w_pseudo=0.30; Rules 1–3 |
| Classes | 0 bg, 1 stroma, 2 benign, 3 G3, 4 G4, 5 G5 |
| **Teacher / external report** | `…/h200x4/epoch_042_cancer_0.7420.pth` (PANDA+ cancer **0.554**) |
| Pseudo Round 1 | Job `5307779`; tag `pseudo_r1_isup_wmfix`; ISUP full-weight edit; wide_margin_unresolved (no G3/G4 hard rewrite); BN batch-stats; freeze 5; backbone LR×0.05 |
| Bias fallback | `round_control.apply_bias_fallback`; leak tolerance **1.05** (tightened from 1.10 after full-weight redesign); cancer/g5 any decline |
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
