#!/usr/bin/env bash
#SBATCH --job-name=oeem_probe
#SBATCH --partition=gpu
#SBATCH -o logs/oeem_probe_%j.out
#SBATCH -e logs/oeem_probe_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:vendor/TRIDENT
export TORCH_CUDNN_ENABLED=0
export OMP_NUM_THREADS=1
python scripts/probe_oeem_real_batches.py --batches 80 --batch-size 4
