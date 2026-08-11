# PANDA Gleason segmentation (UNI2-h + UPerNet)

HPC project for **pixel-level Gleason grading** on Radboud PANDA WSIs, with external check on **PANDA+**, ISUP diagnostics, and planned **pseudo-label / ISUP consistency** work.

**Primary contact use of this README:** hand off to Dr. Omar for designing the **in-loss pseudo-label / ISUP feedback loop**.

**Day-by-day status (Omar):** open [`DAILY_PROGRESS.md`](DAILY_PROGRESS.md) in this folder — newest date at the top.

---

## Where things live (HPC)

| Path | What |
|------|------|
| `/common/omarmlab/members/anh/panda_project/` | Code, scripts, checkpoints, logs, docs |
| `DAILY_PROGRESS.md` | **Omar day log** (mirrored under `outputs/` / `outputs/docs/`) |
| `/common/omarmlab/members/anh/panda_data/` | Raw `slides/`, `masks/`, downloads |
| `outputs/kept_extract/raw/` | Production kept patches (H5, ~432G) |
| `outputs/pen_filter_v33/` | Per-slide keep/drop decisions (tag **v33**) |
| `outputs/splits/` | `panda_{train,val,test}.csv` (~440k / 56k / 54k) |
| `outputs/checkpoints/` | Model C (UNI2) + EfficientNet baselines |
| `outputs/evaluation/` | PANDA+ / val / test metrics CSVs |
| `outputs/pseudo_label/` | ISUP diagnostic + (partial) corrected soft labels |
| `outputs/docs/` | Living narrative / daily / experiment logs |
| `assets/ckpts/uni2-h/` | UNI2-h pretrained weights |
| `.env.hpc` | `PANDA_PROJECT`, `PANDA_DATA_ROOT` |

Env: `module load miniconda3/23.11.0-2 && conda activate wsi_seg`  
GPU jobs: partition `gpu` (H200 preferred). Do **not** run long jobs on the login node.

---

## Current best models (Model C = UNI2-h + UPerNet, raw)

| ID | Recipe | Patches/ep | Val cancer | PANDA+ cancer | Checkpoint |
|----|--------|------------|------------|---------------|------------|
| **A (teacher)** | adj soft **0.15** only | 40k | **0.742** | **0.554** | `outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth` |
| B | adj 0.15 + g45 **0.22** | 10k | 0.716 | 0.552 | `…/h200x4_10k_g45soft_fp32/epoch_072_cancer_0.7160.pth` |
| C | continue A, adj 0.15 | 40k | 0.752 | 0.537 | `…/40k_bf16_from742/epoch_029_cancer_0.7521.pth` |
| R4 | adj 0.15 + g45 0.22 (from scratch) | 40k | 0.739 @ep61 | 0.528 (ep35: 0.539) | `…/40k_g45soft_bf16/epoch_061_cancer_0.7394.pth` |
| R5 | adj 0.15 only | 10k | — | — | **failed** (needs resubmit); script `scripts/slurm_train_uni2_10k_adj015_h200x4.sh` |

**Note:** A/B/C were iterative, not a clean factorial ablation. R4/R5 are the cleaner cells. Always prefer **PANDA+** when val and external disagree.

Soft labels (train loss): `src/train/losses.py` → `gleason_adjacent_soft_targets`  
- adj α=0.15: G4 → `0.075 / 0.85 / 0.075` (G3/G4/G5)  
- optional g45 α=0.22: stronger G4↔G5 mix

---

## ISUP diagnostic (Phase 2 measurement)

```bash
# already produced:
outputs/pseudo_label/diagnostic_report.csv          # 5% gate (canonical)
outputs/pseudo_label/diagnostic_report_summary.json
```

Teacher A on **3746 train slides**: overall match **53.9%** (`min_area_pct=0.05`).

| Meta ISUP | Match | Precision of *derived* ISUP | Notes |
|-----------|------:|----------------------------:|-------|
| 0 | 17.8% | 97.8% | Under-predicted; skip correction |
| 1 | 77.2% | 54.0% | Over-calls 3+3 |
| 2 | 52.0% | 38.0% | Includes 2↔3 swaps |
| 3 | 46.6% | 43.3% | Includes 2↔3 swaps |
| 4 | 66.6% | 47.2% | Often locked wrong primary |
| 5 | 71.1% | 86.2% | Best calibrated |

**Rule B candidates (pure swaps):** meta2→pred3 = **91/91** (`3+4→4+3`); meta3→pred2 = **158/158** (`4+3→3+4`). Dual-grade mixes, not the pure-3+3 overcall failure mode.

Code: `src/isup_diagnostic.py`

---

## Phase 3 offline label correction (mask pixels → soft `.npy`)

**Idea:** On ISUP-mismatched slides (ISUP≥1), rewrite **wrong-class Gleason mask pixels** into soft training targets. Does **not** replace model prediction maps wholesale; does **not** change slide-level ISUP alone.

| Rule | When | What happens to mask pixels |
|------|------|-----------------------------|
| Skip | match, or metadata ISUP 0 | unchanged |
| **B soft** | validated 2↔3 swap | wrong-class cancer → soft mix over metadata expected grades |
| **C hard** | other mismatches | wrong-class cancer → one-hot **metadata primary** Gleason |

Protected: background (0), stroma (1), and classes already in metadata Gleason pair.  
Outputs: `outputs/pseudo_label/corrected_labels/{slide_id}/{x}_{y}.npy` shape `(512,512,6)`.

```bash
# Apply corrections from 5% diagnostic (STOP before retrain — review summary)
PYTHONPATH=src python scripts/phase3_apply_corrections.py
```

**Status:** script restored; prior run was partial / aborted (~255 slides). Re-run cleanly from v1 diagnostic before trusting labels. Training does **not** yet consume these `.npy` files by default — wiring is next.

---

## Ask for Dr. Omar: pseudo-label / ISUP loop **in the loss**

### What we have today
1. Offline diagnostic (model → derived ISUP vs metadata).
2. Offline Rule B/C soft/hard mask corrections (planned training input).
3. Train-time **adjacent soft Gleason** labels (grade uncertainty, not ISUP).
4. Planned but **not implemented:** every-N-epoch ISUP audit inside training.

### Design questions (please decide with Omar)
1. **Offline-only vs online loop?**  
   - A: Finish Phase 3 corrections → retrain once (simple, auditable).  
   - B: Every 10 epochs: freeze ckpt → diagnostic → update soft targets / loss weights → resume (self-correct; needs alerts).
2. **What enters the loss?**  
   - Corrected soft masks only on mismatched slides?  
   - Extra ISUP consistency term (slide-level) vs pixel CE/Dice only?  
   - Teacher-student (use teacher A logits) vs metadata-gated mask edit?
3. **Safety / audit**  
   - Freeze correction recipe per 10-ep window; write dated `diagnostic_report` + `correction_manifest`.  
   - Alert if match rate drops or new mismatch pattern appears (lesson from silent 5%→3% trial).
4. **ISUP-0**  
   - Keep skip (tiny FP speckles). Optional later: aux “cancer present?” head (Phase 1 stub exists: `scripts/slurm_train_uni2_phase1_cls_h200x4.sh`).

### Suggested first implementation (minimal)
1. Re-run `phase3_apply_corrections.py` on **5%** report → review `correction_summary.json`.  
2. Teach `BaselinePatchDataset` / UNI2 train loop to load `.npy` soft targets when present.  
3. Retrain from teacher A or from scratch with same recipe as A; compare PANDA+ + ISUP precision/recall (not just match rate).  
4. Only then add a 10-ep diagnostic callback (no silent label mutation without a manifest).

---

## Common commands

```bash
cd /common/omarmlab/members/anh/panda_project
source .env.hpc
module load miniconda3/23.11.0-2 && source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:vendor/TRIDENT

# Train (example: R5 ablation)
sbatch scripts/slurm_train_uni2_10k_adj015_h200x4.sh

# PANDA+ eval
sbatch scripts/slurm_eval_uni2_panda_plus.sh outputs/checkpoints/.../best.pth

# ISUP diagnostic
python src/isup_diagnostic.py --checkpoint outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth
```

---

## Docs to read next

| Doc | Purpose |
|-----|---------|
| **`DAILY_PROGRESS.md`** (repo root) | **Omar: day-by-day status** (start here) |
| `outputs/docs/TECH_NARRATIVE.md` | Full pipeline story (QC → patches → train → ISUP) |
| `outputs/docs/EXPERIMENT_LOG.md` | Why → what → result → decision |
| `outputs/docs/PRESENTATION_OUTLINE.md` | Slide outline |

---

## Git hygiene

Critical train/eval sources must stay **tracked**. If files vanish from disk:

```bash
git checkout HEAD -- src/ scripts/slurm_train_uni2*.sh scripts/slurm_eval_uni2*.sh
```

Do not leave large untracked train scripts only on the login node.
