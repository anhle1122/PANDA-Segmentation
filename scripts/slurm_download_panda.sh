#!/usr/bin/env bash
# Submit bulk PANDA download on HPC (~412 GB zip + unzip + Radboud symlinks).
# Usage:
#   cd /common/omarmlab/members/anh/panda_project
#   source .env.hpc
#   sbatch scripts/slurm_download_panda.sh

#SBATCH --job-name=panda_bulk
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

bash scripts/hpc_bulk_download_panda.sh all
