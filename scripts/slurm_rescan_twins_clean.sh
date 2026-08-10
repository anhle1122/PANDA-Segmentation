#!/bin/bash
#SBATCH --job-name=rescan_clean
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc

python src/twin_ledger.py --rebuild

python src/rescan_twins_alive.py \
  --out-dir outputs/docs/slide_duplicates_scan3_clean \
  --render

echo "DONE $(date)"
