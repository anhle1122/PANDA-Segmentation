#!/usr/bin/env bash
# Fresh check: raw MASK pixels → derive_grade() vs clinical ISUP.
# Distinct from diagnostic_report.csv (model preds → 2018/3746 = 53.9%).
# Usage: sbatch scripts/slurm_mask_isup_vs_clinical.sh

#SBATCH --job-name=mask_isup_clin
#SBATCH --partition=defq
#SBATCH -o logs/mask_isup_vs_clinical_%j.out
#SBATCH -e logs/mask_isup_vs_clinical_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || true
mkdir -p logs

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

echo "=== $(date) | MASK ISUP vs CLINICAL (raw masks, not model) ==="
export PYTHONPATH=src
python -u src/mask_isup_vs_clinical.py \
  --split-csv outputs/splits/panda_train.csv \
  --metadata-csv data/train.csv \
  --min-area-pct 0.05 \
  --out-csv outputs/pseudo_label/mask_isup_vs_clinical.csv
echo "Finished: $(date)"
