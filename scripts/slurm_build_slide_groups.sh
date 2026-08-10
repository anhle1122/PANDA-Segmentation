#!/bin/bash
#SBATCH --job-name=slide_groups
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc

python src/build_slide_groups.py --edge-min 0.30 --group-thr 0.50

echo "DONE $(date)"
