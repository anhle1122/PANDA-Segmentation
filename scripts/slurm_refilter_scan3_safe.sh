#!/bin/bash
#SBATCH --job-name=refilter_safe
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc
python src/refilter_scan3_safe.py
echo DONE
