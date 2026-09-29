#!/bin/bash
#SBATCH --job-name=content_fp
#SBATCH --partition=defq
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc
python src/slide_content_fingerprints.py --workers 24
echo "DONE $(date)"
