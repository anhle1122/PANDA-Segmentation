#!/usr/bin/env bash
# AGGC2022 zero-shot eval on L40S (never H200). Subset1/2 only (Akoya scans).
#   sbatch --export=ALL,CKPT=/path/to/epoch.pth,SCALE_MODE=native \
#     scripts/slurm_eval_aggc2022.sh
#SBATCH --job-name=aggc2022_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/eval_aggc2022_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/eval_aggc2022_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs" "${PANDA_PROJECT}/outputs/evaluation/aggc2022"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

AGGC_ROOT="${AGGC_ROOT:-/common/omarmlab/data/AGGC_2022}"
CKPT="${CKPT:-${1:-}}"
AGGC_GROUPS="${AGGC_GROUPS:-Subset1/Test,Subset2/Train,Subset2/Test}"
SCALE_MODE="${SCALE_MODE:-native}"
MAX_SLIDES="${MAX_SLIDES:-0}"
BATCH_TILES="${BATCH_TILES:-8}"
STAIN_NORM="${STAIN_NORM:-0}"
STAIN_METHOD="${STAIN_METHOD:-macenko}"
STAIN_ARGS=()
if [[ "${STAIN_NORM}" == "1" ]]; then
  STAIN_ARGS=(--stain-norm --stain-method "${STAIN_METHOD}")
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: set CKPT= to a .pth (or pass as arg1)"
  exit 1
fi
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; AGGC2022 eval must use L40S"
  exit 1
fi

echo "=== $(date) | AGGC2022 eval | ckpt=${CKPT} groups=${AGGC_GROUPS} scale_mode=${SCALE_MODE} max_slides=${MAX_SLIDES} ==="
nvidia-smi -L || true
STEM="$(basename "${CKPT}" .pth)"
OUT_SUFFIX="${SCALE_MODE}"
if [[ "${STAIN_NORM}" == "1" ]]; then
  OUT_SUFFIX="${SCALE_MODE}_${STAIN_METHOD}norm"
fi
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/evaluation/aggc2022/${STEM}_${OUT_SUFFIX}}"
python -u "${PANDA_CODE_SRC}/evaluate_aggc2022.py" \
  --checkpoint "${CKPT}" \
  --aggc-root "${AGGC_ROOT}" \
  --groups "${AGGC_GROUPS}" \
  --scale-mode "${SCALE_MODE}" \
  --batch-tiles "${BATCH_TILES}" \
  --max-slides "${MAX_SLIDES}" \
  --amp \
  "${STAIN_ARGS[@]}" \
  --out-dir "${OUT_DIR}"
echo "=== OK $(date) | ${OUT_DIR} ==="
