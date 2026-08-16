#!/usr/bin/env bash
# CPU watcher: AUTO_SUBMIT=1 sbatch per-epoch PANDA / PANDA+ Dice+ISUP on L40S
# when a new epoch_*.pth lands. New-epochs-only (no backfill).
# Does not scancel the H200 train. Does not submit teacher packs.
#SBATCH --job-name=opt3_epoch_eval_watch
#SBATCH --partition=defq
#SBATCH -o logs/watch_epoch_eval_%j.out
#SBATCH -e logs/watch_epoch_eval_%j.err
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
echo "=== $(date) | watch epoch eval AUTO_SUBMIT=${AUTO_SUBMIT:-1} ==="
python -u scripts/watch_opt3_epoch_eval.py \
  --config "${PANDA_PROJECT}/scripts/epoch_eval_targets.json" \
  --interval-sec "${INTERVAL_SEC:-60}" \
  --log-file "${WATCH_LOG:-${PANDA_PROJECT}/outputs/pseudo_label/epoch_eval_watcher.log}" \
  $( [[ "${AUTO_SUBMIT:-1}" == "0" ]] && echo --no-auto-submit || echo --auto-submit )
