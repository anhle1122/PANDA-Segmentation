#!/bin/bash
#SBATCH --job-name=plan_split
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc
python src/plan_group_split.py
echo DONE
