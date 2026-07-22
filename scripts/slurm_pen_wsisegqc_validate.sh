#!/usr/bin/env bash
# Step 2: validate pen.pt on known false-positive slides.
# Usage: sbatch scripts/slurm_pen_wsisegqc_validate.sh

#SBATCH --job-name=panda_pen_v2_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

SLIDES=(
  7d581d0082d6ee0a32165b0f8fe216d8
  4502c2c9c9c1041564225b9d8fad13c1
)

echo "=== pen.pt validation (known false positives) ==="
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: $(date)"
export PYTHONPATH=src

python src/scan_pen_marks_wsisegqc.py --slide-ids "${SLIDES[@]}" --no-npz
python src/pen_wsisegqc_preview.py --slide-ids "${SLIDES[@]}"

echo "Finished: $(date)"
