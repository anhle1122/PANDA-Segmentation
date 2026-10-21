#!/usr/bin/env bash
# A few official-test SICAP tiles: tissue, mask, and 6-class prediction.
# Zero-shot R3 epoch 6 unless CKPT is set. L40S only.
#SBATCH --job-name=sicap_pred_vis
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_pred_vis_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_pred_vis_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED=0
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth}"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/evaluation/sicapv2/epoch_006_pred_vs_mask}"
STEM_ARG=()
if [[ -n "${STEMS:-}" ]]; then
  STEM_ARG=(--stems "${STEMS}")
fi
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; this render must use L40S"
  exit 1
fi
echo "=== $(date) | SICAP pred vs mask | ckpt=${CKPT} ==="
nvidia-smi -L || true
python -u "${PANDA_CODE_SRC}/render_sicap_pred_vs_mask.py" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --batch-size 8 \
  --num-each 3 \
  "${STEM_ARG[@]}"
echo "=== OK $(date) | ${OUT_DIR} ==="
