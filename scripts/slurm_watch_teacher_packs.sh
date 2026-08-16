#!/usr/bin/env bash
# CPU watcher: print (or AUTO_SUBMIT=1 sbatch) teacher-pack cache when a new
# epoch_*.pth lands. Does not scancel the H200 train.
#SBATCH --job-name=opt3_teacher_watch
#SBATCH --partition=defq
#SBATCH -o logs/watch_teacher_packs_%j.out
#SBATCH -e logs/watch_teacher_packs_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
echo "=== $(date) | watch teacher packs AUTO_SUBMIT=${AUTO_SUBMIT:-0} ==="
# Default AUTO_SUBMIT=0: detect/enqueue only. New-epochs-only, one L40S queue.
python -u scripts/watch_opt3_teacher_packs.py \
  --config "${PANDA_PROJECT}/scripts/teacher_watch_targets.json" \
  --interval-sec "${INTERVAL_SEC:-60}" \
  $( [[ "${AUTO_SUBMIT:-0}" == "1" ]] && echo --auto-submit || echo --no-auto-submit )
