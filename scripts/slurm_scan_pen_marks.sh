#!/usr/bin/env bash
# Scan clean Radboud slides for pen marks (green/blue/black ink).
# Usage:
#   cd /common/omarmlab/members/anh/panda_project
#   sbatch scripts/slurm_scan_pen_marks.sh
# Resume after interrupt:
#   sbatch scripts/slurm_scan_pen_marks.sh   # uses --resume if checkpoint exists

#SBATCH --job-name=panda_pen
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

RESUME=()
if [[ -f outputs/pen_mark_scan_checkpoint.json ]]; then
  RESUME=(--resume)
  echo "Checkpoint found — resuming pen mark scan"
fi

echo "=== PANDA pen mark scan ==="
echo "  PANDA_DATA_ROOT=$PANDA_DATA_ROOT"
echo "  Started: $(date)"
export PYTHONPATH=src
python src/scan_pen_marks.py --from-clean "${RESUME[@]}" \
  2>&1 | tee -a outputs/pen_mark_scan_green_blue_black.log
echo "Finished: $(date)"
