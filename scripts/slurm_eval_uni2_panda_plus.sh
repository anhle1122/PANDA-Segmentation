#!/usr/bin/env bash
# Evaluate UNI2-h + UPerNet on PANDA+ only.
# Usage: sbatch scripts/slurm_eval_uni2_panda_plus.sh /path/to/checkpoint.pth
# Default: 1× H100 so it doesn't compete with 4×H200 training.
#SBATCH --job-name=uni2_eval_plus
#SBATCH --partition=gpu
#SBATCH -o logs/eval_uni2_panda_plus_%j.out
#SBATCH -e logs/eval_uni2_panda_plus_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/evaluation
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"
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
NGPU="${SLURM_GPUS_ON_NODE:-1}"

CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: pass checkpoint path as arg1"
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

PLUS_SPLIT="${PANDA_PROJECT}/outputs/panda_plus/panda_plus_patches.csv"
PLUS_MASK_DIR="${PANDA_PROJECT}/outputs/panda_plus/masks"
STEM="$(basename "${CKPT}" .pth)"
OUT="outputs/evaluation/uni2_upernet_raw_panda_plus_${STEM}_labeled.csv"

echo "=== $(date) | UNI2 PANDA+ eval (gt>=2, classes 2–5) | ${NGPU}× GPU | ckpt=${CKPT} ==="
echo "OUT=${OUT}"

if [[ "${NGPU}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NGPU}" src/evaluate.py \
    --checkpoint "${CKPT}" \
    --arch uni2_upernet \
    --split "${PLUS_SPLIT}" \
    --mode raw \
    --out "${OUT}" \
    --batch-size 4 \
    --num-workers 4 \
    --amp \
    --allow-missing-h5 \
    --mask-dir "${PLUS_MASK_DIR}" \
    --mask-suffix "_pandaplus_mask.png" \
    --no-prefer-h5-masks \
    --panda-plus-eval
else
  python src/evaluate.py \
    --checkpoint "${CKPT}" \
    --arch uni2_upernet \
    --split "${PLUS_SPLIT}" \
    --mode raw \
    --out "${OUT}" \
    --batch-size 4 \
    --num-workers 4 \
    --amp \
    --allow-missing-h5 \
    --mask-dir "${PLUS_MASK_DIR}" \
    --mask-suffix "_pandaplus_mask.png" \
    --no-prefer-h5-masks \
    --panda-plus-eval
fi

echo "=== EVAL FINISHED $(date) ==="
echo "Report: ${OUT}"
