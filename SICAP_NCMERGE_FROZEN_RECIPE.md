# SICAP NC-merge fine-tune — frozen recipe

**Status:** FROZEN for Val2/Val3/Val4 + eventual official Test (do not change without an explicit new freeze).  
**Frozen date:** 2026-09-30  
**Reference run:** job `6062694`, tag `opt3_sicap_ft_ncmerge_val1`, commit of trainer stack `988379e` (+ later log-only commits).  
**Best Val1 ckpt (selection rule below):** `outputs/checkpoints/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val1/epoch_004_cancer_0.6122.pth`

Entry point (unchanged):

```bash
sbatch --job-name=sicap_ncmerge_h200 --gres=gpu:h200:2 \
  --export=ALL,RUN_TAG=opt3_sicap_ft_ncmerge_valN,FOLD=N,EPOCHS=30,EARLY_STOP_PATIENCE=10 \
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

Patient-level separation audited 2026-09-30 (see § Patient audit). **Do not** score nc-merge on `partition/Test/Test.xlsx` until Val2–4 finish and the user confirms freeze + epoch policy.

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

## Epoch selection + early stop (frozen)

| Rule | Value |
|---|---|
| Max epochs | **30** |
| Early stop | **patience 10** on **in-loop honest Val{N} Test cancer Dice** (full CM; remap pred {1,2}→0; NC not ignored) |
| Selected ckpt | Named `epoch_XXX_cancer_Y.pth` with **best** honest Val cancer Dice (not `best.pth` alone; not PANDA+; not inflated ignore-0 Dice) |
| Secondary log | Watcher PANDA+ clean-30 + ISUP for drift; **does not** override Val selection |
| Headline (later) | Once: `partition/Test/Test.xlsx` on the **selected** ckpt per fold or agreed ensemble — **hold until user confirms** |

Val1 result under this rule: early-stop at ep14; **selected ep4** (cancer 0.6122).

---

## Eval protocol (reporting)

- Pixel: `evaluate_sicapv2.py` — NC/G3/G4/G5, mean4, cancer, binary, nc_to_cancer.
- Slide ISUP: same script — `derive_grade(pred pixel counts)` vs `wsi_labels` ISUP; write `sicapv2_slide_isup.csv`.
- PANDA+: optional watcher; domain check only.

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
