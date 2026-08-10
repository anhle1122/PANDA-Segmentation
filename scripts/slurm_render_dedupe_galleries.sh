#!/usr/bin/env bash
# Render all dedupe review galleries (545 safe + 55 multi + lower IoU).
#SBATCH --job-name=dedupe_gallery
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:${PYTHONPATH:-}

echo "=== render dedupe galleries === $(date) host=$(hostname)"
python src/render_dedupe_galleries.py
echo "Finished: $(date)"
echo "Open: outputs/docs/slide_duplicates/galleries/index.html"
