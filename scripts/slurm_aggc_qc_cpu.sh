#!/usr/bin/env bash
# CPU: empty-tile skip vs annotations + Macenko thumbnail stain vs tissue.
#SBATCH --job-name=aggc_skip_stain_qc
#SBATCH --partition=defq
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/aggc_qc_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/aggc_qc_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs" "${PANDA_PROJECT}/outputs/evaluation/aggc2022/qc_skip_stain"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
GROUPS="${AGGC_GROUPS:-Subset1/Test,Subset2/Test,Subset3/Test}"
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/evaluation/aggc2022/qc_skip_stain}"
echo "=== $(date) | skip+stain QC groups=${GROUPS} ==="
python -u "${PANDA_PROJECT}/scripts/aggc_skip_stain_qc.py" \
  --groups "${GROUPS}" \
  --out-dir "${OUT}"
echo "=== OK $(date) | ${OUT} ==="
