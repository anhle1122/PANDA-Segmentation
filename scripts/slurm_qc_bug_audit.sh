#!/usr/bin/env bash
# Grade-mismatch QC bug audit (L0 re-read of failed slides).
# Usage: sbatch scripts/slurm_qc_bug_audit.sh

#SBATCH --job-name=panda_audit
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || true

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

echo "=== QC grade-mismatch bug audit ==="
echo "  Started: $(date)"
export PYTHONPATH=src
python src/qc_bug_audit_grade_mismatch.py
echo "Finished: $(date)"
