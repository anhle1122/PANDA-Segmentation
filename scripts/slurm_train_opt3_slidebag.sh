#!/usr/bin/env bash
# Option 3: slide-bag training + dual ISUP losses
#   L = CE+Dice + λ_slide * derived_ISUP_from_seg + λ_grade * grade_head
# Args: [epochs] [resume_checkpoint]
# Override GPU count at submit time, e.g.:
#   sbatch --gres=gpu:h200:2 scripts/slurm_train_opt3_slidebag.sh
#   sbatch --gres=gpu:h200:4 scripts/slurm_train_opt3_slidebag.sh
# Auto-resumes from outputs/checkpoints/.../latest.pth when present (unless $2 set).
#SBATCH --job-name=opt3_slidebag
# H200 currently needs preemptable + part_preemptable (gpu+normal →
# ReqNodeNotAvail on cp095-098). Preemptable ⇒ requeue + auto-resume.
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH -o logs/train_opt3_slidebag_%j.out
#SBATCH -e logs/train_opt3_slidebag_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
# 64G OOMed @ ~41min (val). Bags stack up to 323 patches/slide in RAM.
# Modest share still OK at 200G on mixed H200 nodes; 4-GPU override higher.
#   sbatch --gres=gpu:h200:4 --mem=400G --cpus-per-task=16 ...
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:2
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

EPOCHS="${1:-100}"
RESUME="${2:-}"
NGPU="${SLURM_GPUS_ON_NODE:-${SLURM_JOB_NUM_GPUS:-2}}"
RUN_TAG="${RUN_TAG:-pseudo_r1_opt3_slidebag}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
LAMBDA_SLIDE="${LAMBDA_SLIDE:-0.3}"
LAMBDA_GRADE="${LAMBDA_GRADE:-0.3}"
MICRO_BS="${MICRO_BS:-4}"
SLIDES_PER_EPOCH="${SLIDES_PER_EPOCH:-256}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-5}"
MAX_VAL_PATCHES="${MAX_VAL_PATCHES:-20000}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
LATEST="${CKPT_DIR}/latest.pth"

# Prefer explicit $2; else auto-resume so 2↔4 GPU upgrades never restart from scratch.
if [[ -z "${RESUME}" && -f "${LATEST}" ]]; then
  RESUME="${LATEST}"
  echo "Auto-resume from ${RESUME}"
fi

echo "=== $(date) | OPTION 3 slide-bag | ${NGPU}x H200 | tag=${RUN_TAG} ==="
echo "λ_slide=${LAMBDA_SLIDE} λ_grade=${LAMBDA_GRADE} micro_bs=${MICRO_BS} slides/ep=${SLIDES_PER_EPOCH}"
echo "resume=${RESUME:-none}"

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  src/train_uni2_opt3_slidebag.py
  --mode raw
  --run-tag "${RUN_TAG}"
  --epochs "${EPOCHS}"
  --lambda-slide "${LAMBDA_SLIDE}"
  --lambda-grade "${LAMBDA_GRADE}"
  --micro-batch-size "${MICRO_BS}"
  --slides-per-epoch "${SLIDES_PER_EPOCH}"
  --freeze-backbone-epochs "${FREEZE_EPOCHS}"
  --max-val-patches "${MAX_VAL_PATCHES}"
  --adjacent-soft-alpha 0.15
  --grad-clip 1.0
  --num-workers 2
  --amp
  --allow-missing-h5
)
if [[ -n "${UNI2_CKPT}" && -f "${UNI2_CKPT}" ]]; then
  CMD+=(--uni2-checkpoint "${UNI2_CKPT}")
fi
if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

echo "${CMD[*]}"
"${CMD[@]}"

echo "=== $(date) | OPTION 3 training finished ==="
BEST="${CKPT_DIR}/best.pth"
if [[ -f "${BEST}" ]]; then
  echo "Submitting PANDA+ eval on ${BEST}"
  sbatch scripts/slurm_eval_uni2_panda_plus.sh "${BEST}" || true
fi
