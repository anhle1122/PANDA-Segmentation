#!/usr/bin/env bash
# Train one referee ablation. No r2redo sibling takeover.
# Required env: RUN_TAG CORRECTED_DIR
#SBATCH --job-name=opt3_ab
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/train_opt3_ab_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/train_opt3_ab_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:2
#SBATCH --requeue
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_restore/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ ! -f "${PANDA_CODE_SRC}/train_uni2_opt3_slidebag.py" ]]; then
  echo "ERROR: missing ${PANDA_CODE_SRC}/train_uni2_opt3_slidebag.py"
  exit 1
fi
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
NGPU="${SLURM_GPUS_ON_NODE:-${SLURM_JOB_NUM_GPUS:-2}}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
CORRECTED_DIR="${CORRECTED_DIR:?set CORRECTED_DIR}"
SIBLING_L40S="${SIBLING_L40S:-}"
SIBLING_H200="${SIBLING_H200:-}"
# Comma-separated exclusive set (first runner wins; cancels the rest). Prefer this when
# more than L40S/H200 backups exist (e.g. A100).
SIBLING_NAMES="${SIBLING_NAMES:-}"
MY_NAME="${SLURM_JOB_NAME:-opt3_ab}"
if [[ -z "${SIBLING_NAMES}" && -n "${SIBLING_L40S}" && -n "${SIBLING_H200}" ]]; then
  SIBLING_NAMES="${SIBLING_L40S},${SIBLING_H200}"
fi
if [[ -n "${SIBLING_NAMES}" ]]; then
  echo "SIBLING_POLICY me=${SLURM_JOB_ID:-none}:${MY_NAME} names=${SIBLING_NAMES}"
  while read -r jid jst jname; do
    [[ -z "${jid}" ]] && continue
    [[ "${jid}" == "${SLURM_JOB_ID:-}" ]] && continue
    if [[ "${jst}" == "RUNNING" || "${jst}" == "CONFIGURING" || "${jst}" == "COMPLETING" ]]; then
      echo "BACKUP_EXIT sibling already ${jst} job=${jid} name=${jname}"
      exit 0
    fi
    echo "TAKEOVER scancel sibling job=${jid} name=${jname} state=${jst}"
    scancel "${jid}" || true
  done < <(squeue -u "${USER}" -h -n "${SIBLING_NAMES}" -o '%i %T %j' 2>/dev/null || true)
fi

for banned in opt3_omar6_locked opt3_omar6_grouped_soft01 opt3_omar6_round2_ep14ref opt3_omar6_round3_ep7ref opt3_omar6_round4_ep6ref opt3_omar6_round2redo_tau06_benign; do
  if [[ "${RUN_TAG}" == "${banned}" ]]; then
    echo "ERROR: refuse tag ${banned}."
    exit 1
  fi
done
if [[ ! -d "${CORRECTED_DIR}" ]]; then
  echo "ERROR: missing corrected dir ${CORRECTED_DIR}"
  exit 1
fi
if [[ ! -f "${CORRECTED_DIR}/balance_report.json" ]]; then
  echo "ERROR: ${CORRECTED_DIR} incomplete (no balance_report.json)"
  exit 1
fi
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
mkdir -p "${CKPT_DIR}"
RESUME=""
if [[ "${COLD:-0}" == "1" ]]; then
  if compgen -G "${CKPT_DIR}/*.pth" > /dev/null; then
    echo "ERROR: COLD=1 but ${CKPT_DIR} already has weights. Unset COLD to resume."
    exit 1
  fi
  echo "COLD=1 scratch start"
elif [[ -f "${CKPT_DIR}/latest.pth" ]]; then
  RESUME="${CKPT_DIR}/latest.pth"
  echo "AUTO_RESUME ${RESUME}"
elif compgen -G "${CKPT_DIR}/epoch_*.pth" > /dev/null; then
  RESUME="$(ls -1t "${CKPT_DIR}"/epoch_*.pth | head -1)"
  echo "AUTO_RESUME ${RESUME}"
fi

UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
LAMBDA_SLIDE="${LAMBDA_SLIDE:-0.3}"
LAMBDA_GRADE="${LAMBDA_GRADE:-0.3}"
MICRO_BS="${MICRO_BS:-4}"
LIVE_PATCHES="${LIVE_PATCHES:-64}"
LIVE_CHUNK="${LIVE_CHUNK:-4}"
SLIDES_PER_EPOCH="${SLIDES_PER_EPOCH:-256}"

echo "=== $(date) | ablation Opt3 | ${NGPU}x GPU | tag=${RUN_TAG} ==="
echo "LABEL_SOURCE=corrected CORRECTED_DIR=${CORRECTED_DIR}"
echo "CODE_SRC=${PANDA_CODE_SRC}"

CMD=(
  torchrun --standalone --nproc_per_node="${NGPU}"
  "${PANDA_CODE_SRC}/train_uni2_opt3_slidebag.py"
  --mode raw
  --run-tag "${RUN_TAG}"
  --label-source corrected
  --corrected-dir "${CORRECTED_DIR}"
  --epochs "${EPOCHS}"
  --lambda-slide "${LAMBDA_SLIDE}"
  --lambda-grade "${LAMBDA_GRADE}"
  --micro-batch-size "${MICRO_BS}"
  --live-patches "${LIVE_PATCHES}"
  --live-chunk "${LIVE_CHUNK}"
  --slides-per-epoch "${SLIDES_PER_EPOCH}"
  --freeze-backbone-epochs 100
  --decode-norm gn
  --max-val-patches 20000
  --adjacent-soft-alpha 0.1
  --include-benign-soft
  --min-slide-patches 5
  --min-area-pct 0.0
  --ckpt-every-slides 8
  --save-every 1
  --keep-checkpoints 0
  --grad-clip 1.0
  --num-workers 4
  --amp
  --augment
  --allow-missing-h5
  --lora
  --lambda-slide-warmup
  --grad-checkpoint
  --decoder-checkpoint
)
if [[ -n "${UNI2_CKPT}" && -f "${UNI2_CKPT}" ]]; then
  CMD+=(--uni2-checkpoint "${UNI2_CKPT}")
fi
if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

echo "${CMD[*]}"
"${CMD[@]}"
echo "=== $(date) | ablation finished ==="
