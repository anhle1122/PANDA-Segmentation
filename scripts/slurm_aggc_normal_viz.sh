#!/usr/bin/env bash
# Overlay stroma-inside-Normal on named AGGC slides (L40S).
#SBATCH --job-name=aggc_normal_viz
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/aggc_normal_viz_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/aggc_normal_viz_%j.err
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
CKPT="${CKPT:?set CKPT}"
SLIDES="${SLIDES:?set SLIDES with +}"
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/evaluation/aggc2022/worst_normal_viz}"
echo "=== $(date) | normal viz ckpt=${CKPT} slides=${SLIDES} ==="
python -u "${PANDA_PROJECT}/scripts/aggc_worst_normal_viz.py" \
  --checkpoint "${CKPT}" \
  --slides "${SLIDES}" \
  --out-dir "${OUT}"
echo "=== OK $(date) | ${OUT} ==="
