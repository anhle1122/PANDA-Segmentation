#!/usr/bin/env bash
# ROUND 1 of iterative pseudo-label self-training.
# Trains a NEW model from scratch (pretrained UNI2-h backbone, fresh decoder)
# with ISUP-informed single loss: Rules 1-3 rewrite flagged pixels in the
# seg target (original mask elsewhere). Correcting rules: rule1_soft_tie,
# rule2_adjacent_invented, rule3_invented_default (~442 slides). Wide-margin
# 2<->3 swaps are wide_margin_unresolved (NO pixel rewrite).
# Original mask files are never modified. Auto-runs PANDA test + PANDA+ eval at
# the end, which is the mandatory gate before any Round 2.
# Args: [epochs] [resume_checkpoint]
#SBATCH --job-name=pseudo_r1
#SBATCH --partition=gpu
#SBATCH -o logs/train_pseudo_round1_%j.out
#SBATCH -e logs/train_pseudo_round1_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:4
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs
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

# Env has cu13 NCCL/cuDNN pollution. Vendor NCCL 2.21.5 + disable cuDNN.
SP="${CONDA_PREFIX}/lib/python3.11/site-packages"
VENDOR_NCCL="${PANDA_PROJECT}/outputs/libs/nccl-cu11"
TORCH_LIB="${SP}/torch/lib"
CUDART11_LIB="${SP}/nvidia/cuda_runtime/lib"
CUBLAS_LIB="${SP}/nvidia/cublas/lib"
if [[ ! -f "${VENDOR_NCCL}/libnccl.so.2" ]]; then
  echo "ERROR: missing ${VENDOR_NCCL}/libnccl.so.2 (NCCL 2.21.5)."
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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

EPOCHS="${1:-100}"
RESUME="${2:-}"
NGPU="${SLURM_GPUS_ON_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM="${ACCUM:-2}"
ADJ_SOFT="${ADJ_SOFT:-0.22}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-10000}"
MAX_VAL_PATCHES="${MAX_VAL_PATCHES:-20000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MIN_EPOCHS="${MIN_EPOCHS:-70}"
PATIENCE="${PATIENCE:-20}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-5}"
BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.05}"
# Fresh tag: wide_margin_unresolved (no G3/G4 hard rewrite) + ISUP single loss + BN fix.
RUN_TAG="${RUN_TAG:-pseudo_r1_isup_wmfix}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
MANIFEST="${MANIFEST:-${PANDA_PROJECT}/outputs/pseudo_label/round1_rule_manifest.csv}"
PRED_DIR="${PRED_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/round1_source_pred}"
W_SEG="${W_SEG:-0.70}"
W_PSEUDO="${W_PSEUDO:-0.30}"

for required in "${MANIFEST}" "${PRED_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: missing pseudo-label input: ${required}"
    exit 1
  fi
done
N_CACHED=$(ls -1 "${PRED_DIR}" | grep -c '_srcpred.h5$' || true)
echo "Source-prediction cache files: ${N_CACHED}"
if [[ "${N_CACHED}" -lt 1 ]]; then
  echo "ERROR: prediction cache is empty; run slurm_cache_source_predictions.sh first."
  exit 1
fi

echo "=== $(date) | PSEUDO-LABEL ROUND 1 | ${NGPU}x H200 | epochs=${EPOCHS} ==="
echo "loss = ISUP-informed segmentation (Rules 1-3 rewrite flagged pixels in seg target)"
echo "base seg_target = ORIGINAL MASK (Round 1) | no fighting dual loss"
echo "decode BN = batch-stats only (track_running_stats=False; matches teacher A)"
echo "freeze=${FREEZE_EPOCHS} ep | backbone_lr_mult=${BACKBONE_LR_MULT}"
echo "manifest=${MANIFEST}"
echo "pred_dir=${PRED_DIR}"
echo "ckpt_dir=${CKPT_DIR}"
echo "eff_batch=$((BATCH_SIZE * NGPU * ACCUM)) | ${PATCHES_PER_EPOCH} train/ep | grad_clip=${GRAD_CLIP}"

python src/make_train_splits.py

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  src/train_uni2_upernet.py
  --mode raw
  --run-tag "${RUN_TAG}"
  --pseudo-label
  --pseudo-manifest "${MANIFEST}"
  --pseudo-pred-dir "${PRED_DIR}"
  --w-seg "${W_SEG}"
  --w-pseudo "${W_PSEUDO}"
  --epochs "${EPOCHS}"
  --min-epochs "${MIN_EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --accum-steps "${ACCUM}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor 2
  --patches-per-epoch "${PATCHES_PER_EPOCH}"
  --max-val-patches "${MAX_VAL_PATCHES}"
  --val-every 1
  --val-batch-size 8
  --save-every 5
  --adjacent-soft-alpha "${ADJ_SOFT}"
  --grad-clip "${GRAD_CLIP}"
  --freeze-backbone-epochs "${FREEZE_EPOCHS}"
  --backbone-lr-mult "${BACKBONE_LR_MULT}"
  --lr 1e-4
  --amp
)
if [[ -n "${UNI2_CKPT}" && -f "${UNI2_CKPT}" ]]; then
  CMD+=(--uni2-checkpoint "${UNI2_CKPT}")
fi
# Round 1 starts a NEW model by design; resume only for restarts of THIS round
# (e.g. moving from 3 to 4 GPUs when one frees up).
if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  echo "Resume checkpoint: ${RESUME}"
  CMD+=(--resume "${RESUME}")
fi
"${CMD[@]}"

echo "=== ROUND 1 TRAINING FINISHED $(date) ==="

BEST="${CKPT_DIR}/best.pth"
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "SKIP_EVAL=1 -- skipping auto-eval (smoke run)."
elif [[ -f "${BEST}" ]]; then
  echo "=== $(date) | MANDATORY Round 1 eval: PANDA test + PANDA+ ==="
  PLUS_SPLIT="${PANDA_PROJECT}/outputs/panda_plus/panda_plus_patches.csv"
  PLUS_MASK_DIR="${PANDA_PROJECT}/outputs/panda_plus/masks"
  python src/evaluate.py \
    --checkpoint "${BEST}" \
    --arch uni2_upernet \
    --split panda_test \
    --mode raw \
    --out "outputs/evaluation/uni2_upernet_${RUN_TAG}_panda_test.csv" \
    --batch-size 4 \
    --num-workers 4 \
    --amp
  python src/evaluate.py \
    --checkpoint "${BEST}" \
    --arch uni2_upernet \
    --split "${PLUS_SPLIT}" \
    --mode raw \
    --out "outputs/evaluation/uni2_upernet_${RUN_TAG}_panda_plus.csv" \
    --batch-size 4 \
    --num-workers 4 \
    --amp \
    --allow-missing-h5 \
    --mask-dir "${PLUS_MASK_DIR}" \
    --mask-suffix "_pandaplus_mask.png" \
    --no-prefer-h5-masks \
    --panda-plus-eval
  echo "=== ROUND 1 EVAL FINISHED $(date) ==="
  echo "GATE before Round 2:"
  echo "  1) Compare PANDA+ cancer_dice / g5_dice / g3→g4 leak vs teacher A (0.554 cancer)."
  echo "  2) Bias-fallback is STILL REQUIRED for Round 2+: if PANDA+ degrades after training"
  echo "     with --seg-target-dir=<prev model preds>, permanently omit --seg-target-dir"
  echo "     (revert base target to original mask). Full-weight ISUP edits make this more"
  echo "     important, not less — uncaught G3→G4 bias is taught at full strength."
  echo "  See src/train/round_control.py :: apply_bias_fallback / bias_too_heavy."
else
  echo "WARNING: no best.pth at ${BEST}; skip auto-eval"
fi
