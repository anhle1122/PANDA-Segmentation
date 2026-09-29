#!/usr/bin/env bash
# SICAPv2 MPP-aligned eval: upsample 10×→~20× (scale=2), tile 512, score vs native GT.
# Usage:
#   sbatch --export=ALL,CKPT=/path/to/epoch_006_....pth,SCALE_FACTOR=2 \
#     scripts/slurm_eval_sicapv2_mpp.sh
#SBATCH --job-name=sicap_mpp2x
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_mpp_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_mpp_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs" "${PANDA_PROJECT}/outputs/evaluation/sicapv2"
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
# Avoid comma FOLDS in --export
FOLDS="${FOLDS:-1 2 3 4}"
FOLDS_CSV="$(echo "${FOLDS}" | tr ' ' ',')"

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: set CKPT= to a .pth"
  exit 1
fi
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; SICAPv2 eval must use L40S"
  exit 1
fi

OUT_DIR="${PANDA_PROJECT}/outputs/evaluation/sicapv2/mpp${SCALE_FACTOR}x_$(basename "${CKPT}" .pth)"
echo "=== $(date) | SICAPv2 MPP-align scale=${SCALE_FACTOR} | ckpt=${CKPT} folds=${FOLDS_CSV} ==="
nvidia-smi -L || true
python -u "${PANDA_CODE_SRC}/evaluate_sicapv2.py" \
  --checkpoint "${CKPT}" \
  --sicap-root "${SICAP_ROOT}" \
  --folds "${FOLDS_CSV}" \
  --scale-factor "${SCALE_FACTOR}" \
  --batch-size "${EVAL_BS:-8}" \
  --num-workers 4 \
  --amp \
  --out-dir "${OUT_DIR}"
echo "=== OK $(date) | ${OUT_DIR} ==="
