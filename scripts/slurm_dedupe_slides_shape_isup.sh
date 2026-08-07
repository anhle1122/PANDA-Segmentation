#!/usr/bin/env bash
# Shape+ISUP dedupe (stain-invariant). Backs up splits to *_pre_dedupe.csv then drops twins.
# Usage:
#   sbatch scripts/slurm_dedupe_slides_shape_isup.sh           # match + APPLY
#   sbatch scripts/slurm_dedupe_slides_shape_isup.sh --dry-run # match only / dry apply

#SBATCH --job-name=slide_dedupe
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:${PYTHONPATH:-}

echo "=== shape+ISUP dedupe === $(date) host=$(hostname)"
MODE="${1:-}"
if [[ "${MODE}" == "--dry-run" ]]; then
  python src/dedupe_slides_shape_isup.py --iou-min 0.30 --top-k 3 --apply --dry-run
else
  python src/dedupe_slides_shape_isup.py --iou-min 0.30 --top-k 3 --apply
fi
echo "Finished: $(date)"
