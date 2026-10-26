#!/usr/bin/env bash
# Manuscript SICAPv2 example figure (H&E / provided mask / prediction). L40S only.
#SBATCH --job-name=sicap_fig
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_fig_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_fig_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED=0
export PYTHONUNBUFFERED=1
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; this figure job must use L40S"
  exit 1
fi

CKPT="${CKPT:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_opt3_sicap_ft_ncmerge_replay03_val2/epoch_005_cancer_0.6914.pth}"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/figures/sicap_best_examples_replay03_val2_ep005}"
python -u src/figure_sicap_best_examples.py --checkpoint "${CKPT}" --out-dir "${OUT_DIR}"
echo "=== OK $(date) | ${OUT_DIR} ==="
