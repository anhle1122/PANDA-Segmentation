#!/usr/bin/env bash
# Level-0 vs pyramid mask balance verification (30-slide sample).
# Usage: sbatch scripts/slurm_verify_mask_balance.sh

#SBATCH --job-name=panda_bal
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || true

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

echo "=== Mask balance verification ==="
echo "  Started: $(date)"
export PYTHONPATH=src
python src/verify_mask_balance_pyramid.py
echo "Finished: $(date)"
