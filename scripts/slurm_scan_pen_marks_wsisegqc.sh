#!/usr/bin/env bash
# Full clean-set pen detection via wsisegqc pen.pt (Step 3).
# Usage:
#   sbatch scripts/slurm_scan_pen_marks_wsisegqc.sh
# Resume:
#   sbatch scripts/slurm_scan_pen_marks_wsisegqc.sh  # auto --resume if checkpoint

#SBATCH --job-name=panda_pen_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

RESUME=()
if [[ -f outputs/pen_mark_detection_v2/pen_wsisegqc_checkpoint.json ]]; then
  RESUME=(--resume)
  echo "Checkpoint found — resuming pen.pt scan"
fi

echo "=== PANDA pen.pt scan (wsisegqc) ==="
echo "  PANDA_DATA_ROOT=$PANDA_DATA_ROOT"
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: $(date)"
export PYTHONPATH=src
python src/scan_pen_marks_wsisegqc.py --from-clean "${RESUME[@]}" \
  2>&1 | tee -a outputs/pen_mark_detection_v2/scan.log
echo "Finished: $(date)"
