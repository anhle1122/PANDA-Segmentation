# SICAP NC-merge fine-tune — frozen recipe

**Status:** FROZEN for Val2/Val3/Val4 + eventual official Test (do not change without an explicit new freeze).  
**Frozen date:** 2026-09-30
**Early-stop amendment (2026-09-30):** patience **20** (was 10). Val1 job 6062694 already finished under patience 10 (selected ep4); Val2–4 use patience 20. Max epochs still 30.


**Ablation (not a freeze change), 2026-09-30:** Val1 pixel-only `opt3_sicap_ft_ncmerge_val1_pixelonly` with λ_slide=λ_grade=0, patience 20, job **6072948**. CV folds keep λ=0.3. Compare to ep4 under same selection rule.
**Val1 re-run (2026-09-30):** patience-20 fold-1 launched as tag `opt3_sicap_ft_ncmerge_val1_p20` (job **6071738**) so all four CV folds share the same early-stop rule. Historical patience-10 tag `opt3_sicap_ft_ncmerge_val1` (ep4) kept on disk for comparison; **selection for the final CV uses `val1_p20`**, not the old tag.  
**Reference / recipe provenance:** job `6062694` tag `opt3_sicap_ft_ncmerge_val1` (patience 10). **Canonical Val1:** `opt3_sicap_ft_ncmerge_val1_p20`, commit of trainer stack `988379e` (+ later log-only commits).  
**Best Val1 ckpt (selection rule below):** `outputs/checkpoints/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val1/epoch_004_cancer_0.6122.pth`

Entry point (unchanged):

```bash
sbatch --job-name=sicap_ncmerge_h200 --gres=gpu:h200:2 \
  --export=ALL,RUN_TAG=opt3_sicap_ft_ncmerge_valN,FOLD=N,EPOCHS=30,EARLY_STOP_PATIENCE=20 \
  scripts/slurm_train_sicap_ncmerge.sh
```

(`N` ∈ {2,3,4}; Val1 already done.)

---

## Init / model

| Item | Value |
|---|---|
| Init ckpt | `uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth` |
| Arch | UNI2-h + UPerNet + LoRA QKV (r=8, α=16) + ISUP grade head |
| Freeze | `--freeze-backbone-epochs 100` → UNI2 **base frozen** for entire ≤30-ep run |
| Trainable | LoRA A/B + decoder + grade head (`lora_params_in_optimizer>0` required) |
| Decode norm | `--decode-norm gn` |
| AMP / clip | `--amp --grad-clip 1.0` |
| Checkpointing | `--grad-checkpoint --decoder-checkpoint` |
| GPUs | 2×H200 (preferred; match Val1). L40S only if H200 unavailable (note hardware caveat). |

---

## Data / split

| Item | Value |
|---|---|
| Root | `/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2` |
| Train | `partition/Validation/Val{N}/Train.xlsx` |
| Val (selection) | `partition/Validation/Val{N}/Test.xlsx` — **honest** pixel CM |
| Magnification | Native 10× (`scale_factor=1.0`); no MPP×2 for selection |
| Slide ISUP GT | `wsi_labels.xlsx` Gleason → ISUP (bag loss + eval) |
| Masks | On-disk `{0,3,4,5}`; `0` = non-cancer (not PANDA ignore) |

Patient-level separation audited 2026-09-30 (see § Patient audit).  
**Official Test** (`partition/Test/Test.xlsx`) is reserved for Stage C only — never for selection or early stop.  
**Full Train** (`partition/Test/Train.xlsx`, 9959 patches) is Stage B only — after CV fixes epoch count E.

---

## Loss (3 terms, 2 heads)

`total ≈ L_pixel + λ_slide·L_slide + λ_grade·L_grade`

### L_pixel — `--sicap-nc-merge` (required)

- **CE (merged NC):** GT≤2 → `-log(p0+p1+p2)`; GT∈{3,4,5} → adjacent soft CE α=`0.1` among **G3↔G4↔G5 only** (`--no-include-benign-soft`).
- **Dice:** soft Dice on G3, G4, G5 (**all pixels**; cancer-on-NC = FP) + merged NC `p0+p1+p2`.
- Combine `0.5·CE + 0.5·Dice`. **No `ignore_index=0`.**

### L_slide / L_grade

- λ_slide = `0.3` with `--lambda-slide-warmup` (0 ep1–2, ramp, full by ep6).
- λ_grade = `0.3`.
- Slide bag: **live=ALL** patches, `--live-chunk 4`, `--min-slide-patches 5`, `--min-area-pct 0`.
- Micro-batch pixel path: `--micro-batch-size 4`.
- Augment: `--augment`.

Abort / log gates at start: `WIRING_OK … sicap_nc_merge=1 include_benign_soft=0`, `TRAINABLE decoder=… lora_in_optim=…`.

---

## LR / optimizer / schedule

Same as Val1 / Opt3 SICAP FT defaults in `train_uni2_sicap_finetune.py` (AdamW on LoRA+decoder+grade; cosine over `EPOCHS`). Do not change LR, WD, or T_max when launching Val2–4.

---

## Epoch selection + early stop (CV folds only)

| Rule | Value |
|---|---|
| Max epochs | **30** |
| Early stop | **patience 20** on **in-loop honest Val{N} Test cancer Dice** (full CM; remap pred {1,2}→0; NC not ignored) |
| Selected ckpt (per fold) | Named `epoch_XXX_cancer_Y.pth` with **best** honest Val cancer Dice (not `best.pth` alone; not PANDA+; not inflated ignore-0 Dice) |
| Secondary log | Watcher PANDA+ clean-30 + ISUP for drift; **does not** override Val selection |

Val1 historical (`opt3_sicap_ft_ncmerge_val1`, patience 10): ep4 cancer 0.6122. **Canonical Val1 for CV:** `opt3_sicap_ft_ncmerge_val1_p20` (patience 20), same as Val2–4.

---

## Final model (after 4-fold CV) — frozen intent

**Why:** Val{N} Train is only ~95 slides / ~7.5k patches; data is the likely bottleneck.  
`partition/Test/Train.xlsx` = **9,959 patches / 124 slides / 74 patients** = Val{i}Train ∪ Val{i}Test for any fold (~+2.5k patches and more patients vs Val1 Train alone).  
That pool has **no held-out val left** (official Test is disjoint). So:

| Stage | What | Val / stop | Fair check |
|---|---|---|---|
| **A — CV (running)** | Val1_p20 + Val2–4, frozen recipe | Honest Val Test; patience 20 | Per-fold metrics only |
| **B — full Train** | One FT on **all** `partition/Test/Train.xlsx` | **No val.** Fixed **E** epochs, **no early stop** | None (must not peek Test) |
| **C — headline** | Score Stage-B ckpt **once** on `partition/Test/Test.xlsx` | — | Official holdout (21 patients) |
| **C′ — fold holdout (optional, decided a priori)** | Each fold’s **Val-selected** ckpt once on official Test | — | Mean±spread on true holdout; **never** every epoch |

**Selection never uses PANDA+.** Early-stop / best epoch = honest Val{N} Test cancer Dice only (in-loop). PANDA+ every epoch = drift curve only. Official Test is not a selection set.

### How E is chosen (frozen)

After all four CV folds finish:

1. For each fold, take the **best-epoch index** under the honest Val cancer-Dice rule (same as ckpt selection).
2. Set **E = median** of those four integers (if .5, round **up**).
3. Record `(e1,e2,e3,e4) → E` in the run folder + `DAILY_PROGRESS` **before** launching Stage B.
4. Stage B: same loss/LR/freeze/init as CV; `EPOCHS=E`; early-stop **off**; cosine `T_max=E`; tag e.g. `opt3_sicap_ft_ncmerge_fulltrain_e{E}`. Save every epoch; **report epoch E** (last), not a val-picked middle epoch.

**Do not** launch Stage B/C until CV finishes and the user confirms E.  
**Do not** use official Test for model selection or early stopping.

### Code note

Trainer currently only loads `Validation/Val{N}/{Train,Test}.xlsx`. Stage B needs a `--sicap-split full_train` (or equivalent) path over `partition/Test/Train.xlsx` before submit — implement when CV is done, not before.

---

## Scoring policy (frozen — do not use holdout for selection)

| What | Which epochs | Role |
|---|---|---|
| **Val{N} Test** (fold’s own held-out) | **Every epoch** (in-loop + watcher `evaluate_sicapv2`) | **Only** selection / early-stop metric = honest cancer Dice |
| **PANDA+** (leak-split) | **Every epoch** (watcher) | Drift curve only — **never** pick epoch from PANDA+ |
| **Other folds’ Tests** | Never for fold N’s model | Those patches are in fold N’s train set |
| **Official `Test/Test.xlsx`** | **Selected ckpt only**, once | Holdout. Pre-declared (before looking): after CV, score each fold’s **selected** epoch once for mean±spread; Stage-B full-Train model also scored once. **No** per-epoch official sweeps. |

**Selection was never PANDA+-driven** (Val1 historical + this CV: early-stop / best = Val cancer Dice only).

Per fold report: selected-epoch Val{N} Dice (+ classes, ISUP) and that same epoch’s PANDA+ as drift. Across folds: mean±spread of the four Val selected numbers. Per-epoch Val/PANDA+ curves = supplementary figure only.

Watchers already on each tag (`sicap_ncm_w_v*`) submit Val{N} + PANDA+ per new `epoch_*.pth`. No extra watcher needed. They do **not** submit official Test.

---

## Patient audit (2026-09-30)

Unit of separation: `wsi_labels.patient_id` (95 patients, 155 slides). Patch → slide via `stem.split("_Block_")[0]`.

| Check | Result |
|---|---|
| `partition/Test/Train.xlsx` ∩ `partition/Test/Test.xlsx` (slides) | **0** |
| Same, **patient_id** | **0** (74 vs 21 patients) |
| Val{i} Train ∩ Test (slides / patients) | **0** for i=1..4 |
| Val{i} Test ∩ Val{j} Test (slides / patients), i≠j | **0** |
| Union Val Tests == Test/Train | **Yes** (124 slides / 74 patients) |
| Any Val Train/Test ∩ official Test patients | **0** |
| Patients with slides in >1 Val Test fold | **0** |

Script: `scripts/verify_sicap_patient_splits.py`.

---

## Out of scope for this freeze

- PANDA `ignore_index=0` / benign↔grade soft (ablation tag only).
- MPP×2 selection.
- SICAP ISUP-referee correction loop (not run; separate proposal).
- Changing Val1 selected epoch after the fact without a new freeze note.
