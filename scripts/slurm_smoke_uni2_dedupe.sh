#!/bin/bash
#SBATCH --job-name=smoke_uni2_dd
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc

python src/smoke_uni2_dedupe.py --n-pairs 40 --n-random 150 --n-patch 24

echo "DONE $(date)"
