#!/usr/bin/env bash
# SICAPv2 fine-tune from R3 ep6 on H200. live=ALL slide ISUP. Never overwrites R3 dir.
# Usage:
#   sbatch --job-name=sicap_ft_h200 --gres=gpu:h200:2 \
#     --export=ALL,RUN_TAG=opt3_sicap_ft_r3ep6_val1_liveall,FOLD=1,EPOCHS=30 \
#     scripts/slurm_train_sicap_finetune.sh
#SBATCH --job-name=sicap_ft_h200
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_ft_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_ft_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:2
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

# CODE_SRC pins a resumed run to the code snapshot it started on.
if [[ -n "${CODE_SRC:-}" ]]; then
  [[ -f "${CODE_SRC}/train_uni2_sicap_finetune.py" ]] || { echo "ERROR: CODE_SRC=${CODE_SRC} missing trainer"; exit 1; }
  export PANDA_CODE_SRC="${CODE_SRC}"
elif [[ -f "${PANDA_PROJECT}/src/train_uni2_sicap_finetune.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
elif [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/src/train_uni2_sicap_finetune.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
else
  echo "ERROR: missing train_uni2_sicap_finetune.py"
  exit 1
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"
export PYTHONUNBUFFERED=1

SP="${CONDA_PREFIX}/lib/python3.11/site-packages"
VENDOR_NCCL="${PANDA_PROJECT}/outputs/libs/nccl-cu11"
TORCH_LIB="${SP}/torch/lib"
CUDART11_LIB="${SP}/nvidia/cuda_runtime/lib"
CUBLAS_LIB="${SP}/nvidia/cublas/lib"
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
if [[ -f "${VENDOR_NCCL}/libnccl.so.2" ]]; then
  export LD_PRELOAD="${VENDOR_NCCL}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

RUN_TAG="${RUN_TAG:-opt3_sicap_ft_r3ep6_val1_liveall}"
FOLD="${FOLD:-1}"
EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
NGPU="${SLURM_GPUS_ON_NODE:-${SLURM_JOB_NUM_GPUS:-2}}"
INIT_CKPT="${INIT_CKPT:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth}"
SICAP_ROOT="${SICAP_ROOT:-/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"

for banned in opt3_omar6_locked opt3_omar6_grouped_soft01 opt3_omar6_round2_ep14ref opt3_omar6_round3_ep7ref opt3_omar6_round4_ep6ref; do
  if [[ "${RUN_TAG}" == "${banned}" ]]; then
    echo "ERROR: refuse overwriting live/R-series tag ${banned}"
    exit 1
  fi
done
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "ERROR: missing INIT_CKPT=${INIT_CKPT}"
  exit 1
fi
mkdir -p "${CKPT_DIR}"

echo "=== $(date) | SICAP FT H200 | tag=${RUN_TAG} fold=${FOLD} ngpu=${NGPU} ==="
echo "INIT=${INIT_CKPT}"
echo "SRC=${PANDA_CODE_SRC}"
echo "EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE}"
nvidia-smi -L || true

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  "${PANDA_CODE_SRC}/train_uni2_sicap_finetune.py"
  --mode raw
  --run-tag "${RUN_TAG}"
  --sicap-root "${SICAP_ROOT}"
  --fold "${FOLD}"
  --epochs "${EPOCHS}"
  --init-checkpoint "${INIT_CKPT}"
  --micro-batch-size 4
  --live-chunk 4
  --lambda-slide 0.3
  --lambda-grade 0.3
  --lambda-slide-warmup
  --min-slide-patches 5
  --min-area-pct 0.0
  --adjacent-soft-alpha 0.1
  --include-benign-soft
  --decode-norm gn
  --lora
  --grad-checkpoint
  --decoder-checkpoint
  --freeze-backbone-epochs 100
  --amp
  --augment
  --num-workers 4
  --grad-clip 1.0
)
if [[ -n "${UNI2_CKPT}" && -f "${UNI2_CKPT}" ]]; then
  CMD+=(--uni2-checkpoint "${UNI2_CKPT}")
fi
if grep -q -- "--early-stop-patience" "${PANDA_CODE_SRC}/train_uni2_sicap_finetune.py"; then
  CMD+=(--early-stop-patience "${EARLY_STOP_PATIENCE}")
fi
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "ERROR: RESUME=${RESUME} not found"; exit 1; }
  CMD+=(--resume "${RESUME}")
fi
echo "${CMD[*]}"
"${CMD[@]}"
echo "=== $(date) | SICAP FT finished ==="
