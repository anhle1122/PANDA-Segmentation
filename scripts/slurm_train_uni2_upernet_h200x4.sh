#!/usr/bin/env bash
# Model C — UNI2-h + UPerNet, 4× H200 — quality recipe (cp098 often free as 4× idle).
# Loss: 0.5 weighted CE + 0.5 soft Dice with adjacent G3↔G4↔G5 soft labels.
# Defaults: 10k patches/ep, adjacent_soft α=0.22, grad-clip=1.0.
# Args: [epochs] [resume_checkpoint]
#SBATCH --job-name=uni2_h200x4
#SBATCH --partition=gpu
#SBATCH -o logs/uni2_upernet_h200x4_%j.out
#SBATCH -e logs/uni2_upernet_h200x4_%j.err
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
# Preload matched NCCL (env file is 2.29.7). cuDNN stays off so preload doesn't brick convs.
export LD_PRELOAD="${VENDOR_NCCL}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
# Prefer NCCL on H200; fall back: TORCH_DISTRIBUTED_BACKEND=gloo sbatch ...
export TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-nccl}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

EPOCHS="${1:-100}"
RESUME="${2:-}"
NGPU="${SLURM_GPUS_ON_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM="${ACCUM:-2}"
LABEL_SMOOTH="${LABEL_SMOOTH:-0.0}"
# Soften G4↔G5 a bit more than the first run (was 0.15).
ADJ_SOFT="${ADJ_SOFT:-0.22}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-10000}"
MAX_VAL_PATCHES="${MAX_VAL_PATCHES:-20000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MIN_EPOCHS="${MIN_EPOCHS:-70}"
PATIENCE="${PATIENCE:-20}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_h200x4"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
# Prefer healthy ep41 over NaN-corrupted latest.pth from the early-stopped run.
EP41="${CKPT_DIR}/epoch_041_cancer_0.7301.pth"

if [[ -z "${RESUME}" ]]; then
  if [[ -f "${EP41}" ]]; then
    RESUME="${EP41}"
  elif [[ -f "${CKPT_DIR}/best.pth" ]]; then
    RESUME="${CKPT_DIR}/best.pth"
  fi
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${UNI2_CKPT}" ]]; then
  echo "ERROR: Need HF_TOKEN or UNI2_CKPT=path/to/pytorch_model.bin"
  exit 1
fi

echo "=== $(date) | UNI2 quality | ${NGPU}× H200 | 32CPU 500G | epochs=${EPOCHS} ==="
echo "DDP backend=${TORCH_DISTRIBUTED_BACKEND} | cuDNN=${TORCH_CUDNN_ENABLED} | UNI2_CKPT=${UNI2_CKPT}"
echo "${PATCHES_PER_EPOCH} train/ep | ${MAX_VAL_PATCHES} val every epoch | adjacent_soft=${ADJ_SOFT} | no global LS"
echo "aug ON | freeze 3 | accum ${ACCUM} | eff_batch=$((BATCH_SIZE * NGPU * ACCUM)) | grad_clip=${GRAD_CLIP} | save/5 | min_epochs=${MIN_EPOCHS} patience=${PATIENCE}"

python src/make_train_splits.py

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  src/train_uni2_upernet.py
  --mode raw
  --run-tag h200x4
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
  --label-smoothing "${LABEL_SMOOTH}"
  --grad-clip "${GRAD_CLIP}"
  --augment
  --freeze-backbone-epochs 3
  --backbone-lr-mult 0.1
  --lr 1e-4
  --amp
)
if [[ -n "${UNI2_CKPT}" && -f "${UNI2_CKPT}" ]]; then
  CMD+=(--uni2-checkpoint "${UNI2_CKPT}")
fi
if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  echo "Resume checkpoint: ${RESUME}"
  CMD+=(--resume "${RESUME}")
fi
"${CMD[@]}"

echo "=== TRAINING FINISHED $(date) ==="

BEST="${CKPT_DIR}/best.pth"
if [[ -f "${BEST}" ]]; then
  echo "=== $(date) | auto-eval UNI2 best.pth on PANDA test + PANDA+ ==="
  PLUS_SPLIT="${PANDA_PROJECT}/outputs/panda_plus/panda_plus_patches.csv"
  PLUS_MASK_DIR="${PANDA_PROJECT}/outputs/panda_plus/masks"
  python src/evaluate.py \
    --checkpoint "${BEST}" \
    --arch uni2_upernet \
    --split panda_test \
    --mode raw \
    --out outputs/evaluation/uni2_upernet_raw_panda_test.csv \
    --batch-size 4 \
    --num-workers 4 \
    --amp
  python src/evaluate.py \
    --checkpoint "${BEST}" \
    --arch uni2_upernet \
    --split "${PLUS_SPLIT}" \
    --mode raw \
    --out outputs/evaluation/uni2_upernet_raw_panda_plus.csv \
    --batch-size 4 \
    --num-workers 4 \
    --amp \
    --allow-missing-h5 \
    --mask-dir "${PLUS_MASK_DIR}" \
    --mask-suffix "_pandaplus_mask.png" \
    --no-prefer-h5-masks
  echo "=== EVAL FINISHED $(date) ==="
else
  echo "WARNING: no best.pth at ${BEST}; skip auto-eval"
fi
