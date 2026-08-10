#!/usr/bin/env bash
# Rescan twins on currently alive slides (includes prior not-twins).
# Writes outputs/docs/slide_duplicates_rescan_alive/ + galleries. Does NOT apply drops.
# Usage: sbatch scripts/slurm_rescan_twins_alive.sh

#SBATCH --job-name=twins_rescan
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

OUT=outputs/docs/slide_duplicates_rescan_alive
echo "=== twins rescan alive === $(date) host=$(hostname)"
python src/rescan_twins_alive.py --iou-min 0.30 --top-k 3 --out-dir "$OUT" --render
echo "Finished: $(date)"
echo "Open: $OUT/galleries/index.html"
cat "$OUT/rescan_summary.json"
