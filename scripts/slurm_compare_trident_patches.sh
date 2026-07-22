#!/usr/bin/env bash
# Compare naive-grid patch extraction vs Trident-Otsu (15-slide diagnostic).
# Usage:
#   cd /common/omarmlab/members/anh/panda_project
#   sbatch scripts/slurm_compare_trident_patches.sh

#SBATCH --job-name=panda_trident_cmp
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

echo "=== Trident vs naive-grid patch comparison ==="
echo "  Started: $(date)"
export PYTHONPATH=src:vendor/TRIDENT
python src/compare_trident_patch_extraction.py 2>&1 | tee outputs/trident_patch_comparison/run.log
echo "Finished: $(date)"
