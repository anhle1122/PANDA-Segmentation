#!/usr/bin/env bash
# Phase 1 — self-distillation (seg + patch cancer head).
# Smoke: 2 epochs / 200 train patches. Full: omit SMOKE=1.
# Usage:
#   sbatch scripts/slurm_train_uni2_phase1_cls_h200x4.sh          # full 100ep
#   SMOKE=1 sbatch scripts/slurm_train_uni2_phase1_cls_h200x4.sh  # 2ep/200
#SBATCH --job-name=uni2_p1_cls
#SBATCH --partition=gpu
#SBATCH -o logs/uni2_phase1_cls_%j.out
#SBATCH -e logs/uni2_phase1_cls_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:4
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/evaluation
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:vendor/TRIDENT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SP="${CONDA_PREFIX}/lib/python3.11/site-packages"
VENDOR_NCCL="${PANDA_PROJECT}/outputs/libs/nccl-cu11"
TORCH_LIB="${SP}/torch/lib"
CUDART11_LIB="${SP}/nvidia/cuda_runtime/lib"
CUBLAS_LIB="${SP}/nvidia/cublas/lib"
if [[ ! -f "${VENDOR_NCCL}/libnccl.so.2" ]]; then
  echo "ERROR: missing ${VENDOR_NCCL}/libnccl.so.2"
  exit 1
fi
PINNED_LIBS=()
for d in "${VENDOR_NCCL}" "${CUBLAS_LIB}" "${CUDART11_LIB}" "${TORCH_LIB}"; do
  [[ -d "${d}" ]] && PINNED_LIBS+=("${d}")
done
_CLEAN_LD=""
IFS=':' read -r -a _ld_parts <<< "${LD_LIBRARY_PATH:-}"
for p in "${_ld_parts[@]:-}"; do
  [[ -z "${p}" ]] && continue
  [[ "${p}" == *"/nvidia/cu13/"* ]] && continue
  [[ "${p}" == *"/nvidia/nccl/lib"* ]] && continue
  [[ "${p}" == *"/nvidia/cudnn/"* ]] && continue
  _CLEAN_LD="${_CLEAN_LD:+${_CLEAN_LD}:}${p}"
done
export LD_LIBRARY_PATH="$(IFS=:; echo "${PINNED_LIBS[*]}")${_CLEAN_LD:+:${_CLEAN_LD}}"
export LD_PRELOAD="${VENDOR_NCCL}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
export TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-nccl}"
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

SMOKE="${SMOKE:-0}"
NGPU="${SLURM_GPUS_ON_NODE:-4}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
BASELINE_CKPT="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_h200x4/epoch_042_cancer_0.7420.pth"

python src/make_train_splits.py

if [[ "${SMOKE}" == "1" ]]; then
  echo "=== $(date) | Phase 1 SMOKE | 2 epochs / 200 patches | ${NGPU}× H200 ==="
  RUN_TAG="h200x4_phase1_cls_smoke"
  EXTRA=(--epochs 2 --min-epochs 2 --patience 2 --patches-per-epoch 200 --max-val-patches 40 --save-every 1)
else
  echo "=== $(date) | Phase 1 FULL | 100 epochs / 40k | ${NGPU}× H200 ==="
  RUN_TAG="h200x4_phase1_cls"
  EXTRA=(--epochs 100 --min-epochs 80 --patience 20 --patches-per-epoch 40000 --max-val-patches 20000 --save-every 5)
fi

torchrun --standalone --nproc_per_node="${NGPU}" \
  src/train_uni2_upernet.py \
  --mode raw \
  --run-tag "${RUN_TAG}" \
  --batch-size 2 \
  --accum-steps 2 \
  --num-workers 6 \
  --prefetch-factor 2 \
  --val-every 1 \
  --val-batch-size 8 \
  --adjacent-soft-alpha 0.15 \
  --label-smoothing 0.0 \
  --cls-loss-weight 0.1 \
  --tumor-cancer-pct-threshold 0.05 \
  --grad-clip 0.5 \
  --max-nonfinite-batches 3 \
  --augment \
  --freeze-backbone-epochs 3 \
  --backbone-lr-mult 0.1 \
  --lr 1e-4 \
  --amp \
  --amp-dtype bfloat16 \
  --uni2-checkpoint "${UNI2_CKPT}" \
  "${EXTRA[@]}"

echo "=== TRAINING FINISHED $(date) ==="

if [[ "${SMOKE}" == "1" ]]; then
  echo "Smoke only — skip Level-3 full eval. Inspect seg_loss/cls_loss in training_log.csv"
  exit 0
fi

BEST="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}/best.pth"
[[ -f "${BEST}" ]] || { echo "WARNING: no best.pth at ${BEST}"; exit 0; }

echo "=== Level-3 compare vs ${BASELINE_CKPT} ==="
PLUS_SPLIT="${PANDA_PROJECT}/outputs/panda_plus/panda_plus_patches.csv"
PLUS_MASK_DIR="${PANDA_PROJECT}/outputs/panda_plus/masks"
CKPT_STEM="$(basename "${BEST}" .pth)"
OUT_PLUS="outputs/evaluation/uni2_phase1_cls_panda_plus_${CKPT_STEM}_labeled.csv"

torchrun --standalone --nproc_per_node="${NGPU}" src/evaluate.py \
  --checkpoint "${BEST}" \
  --arch uni2_upernet \
  --split panda_val \
  --mode raw \
  --out "outputs/evaluation/uni2_phase1_cls_panda_val.csv" \
  --batch-size 4 \
  --num-workers 4 \
  --amp

torchrun --standalone --nproc_per_node="${NGPU}" src/evaluate.py \
  --checkpoint "${BEST}" \
  --arch uni2_upernet \
  --split "${PLUS_SPLIT}" \
  --mode raw \
  --out "${OUT_PLUS}" \
  --batch-size 4 \
  --num-workers 4 \
  --amp \
  --allow-missing-h5 \
  --mask-dir "${PLUS_MASK_DIR}" \
  --mask-suffix "_pandaplus_mask.png" \
  --no-prefer-h5-masks \
  --panda-plus-eval

python - <<'PY'
import csv, json
from pathlib import Path
base = {
    "val_mean_dice": 0.655,
    "val_cancer_dice": 0.742,
    "val_g5": 0.650,
    "plus_cancer": 0.554,
    "plus_g5": 0.528,
}
log = Path("outputs/checkpoints/uni2_upernet_raw_h200x4_phase1_cls/training_log.csv")
best = None
if log.exists():
    rows = list(csv.DictReader(log.open()))
    scored = [r for r in rows if r.get("cancer_dice")]
    if scored:
        best = max(scored, key=lambda r: float(r["cancer_dice"]))
plus_csv = sorted(Path("outputs/evaluation").glob("uni2_phase1_cls_panda_plus_*_labeled.csv"))
plus = {}
if plus_csv:
    for row in csv.DictReader(plus_csv[-1].open()):
        plus[row["class"]] = row
print("\nLevel-3 comparison table")
print(f"{'metric':28s} {'0.742 best':>12s} {'Phase1 cls':>12s}")
val_mean = float(best["mean_dice"]) if best else float("nan")
val_c = float(best["cancer_dice"]) if best else float("nan")
val_g5 = float(best["dice_g5"]) if best else float("nan")
plus_c = float(plus.get("cancer_dice", {}).get("dice", "nan")) if plus else float("nan")
# cancer_dice is a summary row with dice column
if plus and "cancer_dice" in plus:
    plus_c = float(plus["cancer_dice"]["dice"])
plus_g5 = float(plus["G5"]["dice"]) if plus and "G5" in plus else float("nan")
rows = [
    ("val mean_dice", base["val_mean_dice"], val_mean),
    ("val cancer_dice", base["val_cancer_dice"], val_c),
    ("val G5 dice", base["val_g5"], val_g5),
    ("PANDA+ cancer_dice", base["plus_cancer"], plus_c),
    ("PANDA+ G5 dice", base["plus_g5"], plus_g5),
]
for name, a, b in rows:
    print(f"{name:28s} {a:12.3f} {b:12.3f}")
out = Path("outputs/evaluation/uni2_phase1_cls_level3_compare.json")
out.write_text(json.dumps({
    "baseline_epoch42": base,
    "phase1": {
        "val_mean_dice": val_mean,
        "val_cancer_dice": val_c,
        "val_g5": val_g5,
        "plus_cancer": plus_c,
        "plus_g5": plus_g5,
        "best_epoch": int(float(best["epoch"])) if best else None,
    },
}, indent=2))
print(f"Wrote {out}")
PY

echo "=== PHASE 1 DONE $(date) ==="
