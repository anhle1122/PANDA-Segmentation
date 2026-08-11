#!/bin/bash
#SBATCH --job-name=embed_all_uni2
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --requeue

set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
module load miniconda3/23.11.0-2
source activate wsi_seg
source .env.hpc

python src/embed_all_uni2.py --n-patch 24

echo "DONE $(date)"
