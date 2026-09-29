#!/usr/bin/env bash
# Fair SICAPv2 threshold grid: tune t* on Train, apply on Test (MPP-aligned).
# Usage:
#   sbatch --export=ALL,CKPT=/path/to/epoch_006_....pth,SCALE_FACTOR=2 \
#     scripts/slurm_eval_sicapv2_threshold.sh
#SBATCH --job-name=sicap_thresh
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_thresh_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_thresh_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED=0
export PYTHONUNBUFFERED=1

SICAP_ROOT="${SICAP_ROOT:-/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2}"
CKPT="${CKPT:-${1:-}}"
SCALE_FACTOR="${SCALE_FACTOR:-2}"
TUNE_METRIC="${TUNE_METRIC:-binary_cancer}"
TUNE_MAX="${TUNE_MAX:-2500}"
THRESHOLDS="${THRESHOLDS:-0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85}"
# sbatch --export truncates on commas; allow colon/space separators from the env
THRESHOLDS="$(echo "${THRESHOLDS}" | tr ': ' ',,' | tr -s ',' | sed 's/^,//;s/,$//')"
FOLDS="${FOLDS:-1 2 3 4}"
FOLDS_CSV="$(echo "${FOLDS}" | tr ' ' ',')"

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: set CKPT="
  exit 1
fi
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; use L40S"
  exit 1
fi

OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/evaluation/sicapv2/thresh_mpp${SCALE_FACTOR}x_$(basename "${CKPT}" .pth)}"
echo "=== $(date) | SICAP threshold tune | scale=${SCALE_FACTOR} metric=${TUNE_METRIC} thresholds=${THRESHOLDS} ==="
nvidia-smi -L || true
python -u "${PANDA_CODE_SRC}/evaluate_sicapv2_threshold.py" \
  --checkpoint "${CKPT}" \
  --sicap-root "${SICAP_ROOT}" \
  --folds "${FOLDS_CSV}" \
  --scale-factor "${SCALE_FACTOR}" \
  --thresholds "${THRESHOLDS}" \
  --tune-metric "${TUNE_METRIC}" \
  --tune-max-patches "${TUNE_MAX}" \
  --batch-size 8 \
  --num-workers 4 \
  --amp \
  --out-dir "${OUT_DIR}"
echo "=== OK $(date) | ${OUT_DIR} ==="
