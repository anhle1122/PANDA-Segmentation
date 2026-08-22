#!/usr/bin/env bash
# CPU watcher: AUTO_SUBMIT=1 sbatch per-epoch PANDA / PANDA+ Dice+ISUP
# on H100/A100 (never H200). Backfills missing epochs, up to 4 in parallel.
# Does not scancel the H200 train. Does not submit teacher packs.
#SBATCH --job-name=opt3_epoch_eval_watch
#SBATCH --partition=defq
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_epoch_eval_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_epoch_eval_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs "${PANDA_PROJECT}/outputs/logs"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
MIRROR="${PANDA_PROJECT}/outputs/_code_mirror"
echo "=== $(date) | watch epoch eval AUTO_SUBMIT=${AUTO_SUBMIT:-1} ==="
python -u "${MIRROR}/scripts/watch_opt3_epoch_eval.py" \
  --config "${MIRROR}/scripts/epoch_eval_targets.json" \
  --interval-sec "${INTERVAL_SEC:-60}" \
  --log-file "${WATCH_LOG:-${PANDA_PROJECT}/outputs/pseudo_label/epoch_eval_watcher.log}" \
  $( [[ "${AUTO_SUBMIT:-1}" == "0" ]] && echo --no-auto-submit || echo --auto-submit )
