#!/usr/bin/env bash
# Re-render rescan-alive galleries with NEW / KEPT_NOT_TWIN / MIXED badges.
#SBATCH --job-name=rescan_gallery
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
export PYTHONUNBUFFERED=1

DUP=outputs/docs/slide_duplicates_rescan_alive
OUT="$DUP/galleries"
echo "=== render rescan galleries (NEW vs KEPT_NOT_TWIN) === $(date) host=$(hostname)"
python src/render_dedupe_galleries.py --dup-dir "$DUP" --out-dir "$OUT"
echo "Finished: $(date)"
echo "Open: $OUT/index.html"
echo "Indexes: $OUT/safe_pairs/pair_index_annotated.csv  $OUT/lower_iou/pair_index_annotated.csv"
