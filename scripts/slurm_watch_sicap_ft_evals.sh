#!/usr/bin/env bash
# Watch SICAP FT epochs → PANDA+ + SICAP MPP×2 (L40S evals; never H200).
#SBATCH --job-name=sicap_ft_watch
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_sicap_ft_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_sicap_ft_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export RUN_TAG="${RUN_TAG:-opt3_sicap_ft_r3ep6_val1_liveall}"
export LEAK_PREFIX="${LEAK_PREFIX:-sicap_ft}"
export SICAP_FOLD="${SICAP_FOLD:-1}"
export INTERVAL_SEC="${INTERVAL_SEC:-60}"
export EVAL_BS="${EVAL_BS:-2}"
python -u "${PANDA_PROJECT}/scripts/watch_sicap_ft_evals.py"
