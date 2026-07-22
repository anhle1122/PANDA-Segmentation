#!/usr/bin/env bash
# Run full Radboud mask QC (clean_dataset.py) on HPC.
# Usage:
#   cd /common/omarmlab/members/anh/panda_project
#   sbatch scripts/slurm_clean_dataset.sh

#SBATCH --job-name=panda_qc
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

echo "=== PANDA Radboud QC (clean_dataset.py) ==="
echo "  PANDA_DATA_ROOT=$PANDA_DATA_ROOT"
echo "  Started: $(date)"
python src/clean_dataset.py --local-only --resume --sleep 0
echo "Finished: $(date)"
