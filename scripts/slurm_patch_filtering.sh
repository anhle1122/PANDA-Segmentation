#!/usr/bin/env bash
# HSV patch filtering on Trident coords (discard low-tissue / pen-heavy patches).
# Usage: sbatch scripts/slurm_patch_filtering.sh

#SBATCH --job-name=panda_patch_filt
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:vendor/TRIDENT

RESUME=()
[[ -f outputs/patch_filtering/checkpoint.json ]] && RESUME=(--resume)

echo "=== Patch filtering (Trident coords) ==="
echo "Started: $(date)"
python src/patch_filtering_panda.py --from-clean "${RESUME[@]}" --preview-cap 5 \
  2>&1 | tee -a outputs/patch_filtering/run.log
echo "Finished: $(date)"
