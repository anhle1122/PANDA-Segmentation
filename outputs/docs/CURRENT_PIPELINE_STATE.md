# Current pipeline state

Master record for between-round label correction. Updated **2026-08-21**.

## Selected teacher (do not train Round N+1 until referee is reviewed)

| | |
|--|--|
| **Source model** | Locked r2 **epoch 14** (`opt3_omar6_locked`) |
| **Checkpoint** | `outputs/checkpoints/uni2_upernet_raw_opt3_omar6_locked/epoch_014_cancer_0.3885.pth` |
| **PANDA+ cancer Dice** | **0.642** |
| **PANDA+ ISUP** | **54.2%** |
| **PANDA val cancer Dice** | 0.388 |
| **PANDA val ISUP** | 33.1% |
| **Teacher pack** | `outputs/pseudo_label/teacher_opt3_omar6_locked_locked_r2_ep014/` (3739 `*_srcpred.h5`, `pack_config.json` complete) |
| **Corrections** | `outputs/pseudo_label/corrections_opt3_omar6_locked_locked_r2_ep014/` (2982 H5, job **5513153** COMPLETED) |
| **CORRECTED** | **3.24%** of evaluated pixels (3.07B illegal high-conf swaps). Status: `outputs/docs/opt3_this_run/ep14_referee_status.html` |
| **G5 gate** | Swaps **from** G5 = **25.67%** vs trip line **49.55%** (2× teacher G5 pred share). Flag false. ~24 pp margin. Job used `--no-fail-on-g5-bias`. |
| **Section 4** | Pixel overlap **3.3%** of wmfix flags / **4.3%** of all referee swaps. 78% of referee swaps are on non-wmfix slides. Rule 1 overlap **0.44%**. Job **5514638**. **No train until reviewed.** |
| **GitHub** | `d84135c` on `origin/main` (writable `/tmp/PANDA-Segmentation`; HPC `.git` is RO) |
| **Referee job** | **5513153** (`isup_ref_ep14`, CPU `defq`, τ=0.7; prior **5513100** failed in 15s on mirror data path) |
| **Registry id** | `opt3_omar6_locked_locked_r2_ep014` |
| **Train job** | **5445445** (`opt3_r2_resume9d`) |
| **auto_train** | **false** |

Live ep37 also hit 0.642 PANDA+ cancer / 56.2% ISUP but has **no** teacher pack. Not used.

## Locked r2 run

- Tag `opt3_omar6_locked`. Omar-6 recipe: UNI2-h + UPerNet, LoRA QKV in AdamW, GN, live=64, λ_slide warmup to 0.3.
- Original job **5445276** died on NCCL hang after ep10; resume **5445445** continued the same tag.
- Best PANDA+ cancer among scored locked epochs: **ep14**. Later scored epochs are lower (ep13 0.634, ep11 0.632). Remaining later-epoch evals are not required to pick this teacher.

## Correction rule (current — not Rules 1-3)

Three-way **ISUP referee** (`src/apply_isup_referee.py`), τ **= 0.7**:

1. Agree with expert mask → keep mask.
2. Disagree, `maxprob < 0.7` → ignore (no loss).
3. Disagree, high conf + pred G3/G4/G5 **not** in clinical {primary, secondary} → nearest allowed grade. Legal-grade fights keep the mask.
4. ISUP-0 slides skipped.

Rules 1-3 (`wmfix`, teacher A) are **historical Round 1 only**. They are not the correction path for this round. After the referee finishes, a slide-overlap sanity check vs `round1_rule_manifest.csv` is written next to the corrections dir.

## Historical (kept in registry, not selected)

| model_id | PANDA+ Dice | PANDA+ ISUP | Notes |
|---|---|---|---|
| `opt3_omar6_grouped_soft01_pre_lora_fix_ep029` | 0.609 | 60.4% | Live in-memory trainer; LoRA not in AdamW. Pack + referee already done. |

Incumbent for the Dice gate remains **teacher A ep5** (PANDA+ 0.563).
