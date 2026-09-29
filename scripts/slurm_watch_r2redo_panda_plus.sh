#!/usr/bin/env bash
# Long-running watcher: score Round-2-redo PANDA+ leak-split on L40S as epochs land.
# Never H200. Never scancels train.
#SBATCH --job-name=opt3_r2redo_pp_watch
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_pp_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_pp_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export INTERVAL_SEC="${INTERVAL_SEC:-60}"
python -u "${PANDA_PROJECT}/scripts/watch_r2redo_panda_plus.py"
