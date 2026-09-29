#!/usr/bin/env bash
# Long-running watcher: score one Opt3 AB run's PANDA+ as epoch_*.pth land.
# Required export: RUN_TAG LEAK_PREFIX EVAL_JOB_NAME
# Example:
#   sbatch --job-name=opt3_ab06_pp_watch \
#     --export=ALL,RUN_TAG=opt3_omar6_ab_tau06,LEAK_PREFIX=ab_tau06,EVAL_JOB_NAME=pp_ab06_eval \
#     scripts/slurm_watch_ab_panda_plus.sh
# Never H200. Never scancels train. Eval jobs avoid AB train nodes (cp075/076).
#SBATCH --job-name=opt3_ab_pp_watch
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_ab_pp_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_ab_pp_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
: "${RUN_TAG:?set RUN_TAG}"
: "${LEAK_PREFIX:?set LEAK_PREFIX}"
: "${EVAL_JOB_NAME:?set EVAL_JOB_NAME}"

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export INTERVAL_SEC="${INTERVAL_SEC:-60}"
export EVAL_BS="${EVAL_BS:-2}"

WATCH_PY="${PANDA_PROJECT}/outputs/_code_mirror/scripts/watch_ab_panda_plus.py"
if [[ ! -f "${WATCH_PY}" ]]; then
  WATCH_PY="${PANDA_PROJECT}/scripts/watch_ab_panda_plus.py"
fi
echo "=== $(date) | AB PANDA+ watcher | tag=${RUN_TAG} leak=${LEAK_PREFIX} job=${EVAL_JOB_NAME} ==="
python -u "${WATCH_PY}"
