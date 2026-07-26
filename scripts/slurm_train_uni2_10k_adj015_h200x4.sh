#!/usr/bin/env bash
# Ablation: 10k patches/ep + adjacent soft α=0.15 ONLY (no g45 soft), from scratch → ep100.
# Fair complement to R4 (40k+g45) and teacher A (40k+adj0.15).
# Usage: sbatch [--dependency=...] scripts/slurm_train_uni2_10k_adj015_h200x4.sh
#SBATCH --job-name=uni2_10k_adj015
#SBATCH --partition=gpu
#SBATCH -o logs/uni2_10k_adj015_h200x4_%j.out
#SBATCH -e logs/uni2_10k_adj015_h200x4_%j.err
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
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:vendor/TRIDENT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

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

EPOCHS="${1:-100}"
NGPU="${SLURM_GPUS_ON_NODE:-4}"
RUN_TAG="${RUN_TAG:-h200x4_10k_adj015}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"

MIN_EPOCHS="${MIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-1000}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-10000}"
ADJ_SOFT="${ADJ_SOFT:-0.15}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM="${ACCUM:-2}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_VAL_PATCHES="${MAX_VAL_PATCHES:-20000}"

echo "=== $(date) | UNI2 10k adj-soft=${ADJ_SOFT} ONLY (no g45) | ${NGPU}xH200 | epochs=${EPOCHS} ==="
echo "run_tag=${RUN_TAG} | patches/ep=${PATCHES_PER_EPOCH} | min_epochs=${MIN_EPOCHS} patience=${PATIENCE}"
echo "from scratch | freeze 3 | accum ${ACCUM} | grad_clip=${GRAD_CLIP} | lr=${LR}"
echo "ckpt_dir=${CKPT_DIR}"

if [[ ! -f "${UNI2_CKPT}" ]]; then
  echo "ERROR: UNI2 checkpoint not found: ${UNI2_CKPT}"
  exit 1
fi

python src/make_train_splits.py || true

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  src/train_uni2_upernet.py
  --mode raw
  --run-tag "${RUN_TAG}"
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
  --label-smoothing 0.0
  --grad-clip "${GRAD_CLIP}"
  --augment
  --freeze-backbone-epochs 3
  --backbone-lr-mult 0.1
  --lr "${LR}"
  --amp
  --uni2-checkpoint "${UNI2_CKPT}"
)

echo "CMD: ${CMD[*]}"
"${CMD[@]}"

echo "=== TRAINING FINISHED $(date) ==="
echo "Best/latest under: ${CKPT_DIR}"
