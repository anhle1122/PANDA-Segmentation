#!/usr/bin/env bash
# Watch Stage-B SICAP full-train ckpt dir → PANDA+ drift per epoch, official
# SICAPv2 Test once on the LAST epoch (epoch E).
#   sbatch --export=ALL,RUN_TAG=opt3_sicap_ft_ncmerge_fulltrain_replay03_e6 \
#     scripts/slurm_watch_sicap_stageB.sh
#SBATCH --job-name=sicap_stageB_watch
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_sicap_stageB_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_sicap_stageB_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export RUN_TAG="${RUN_TAG:?set RUN_TAG}"
export LEAK_PREFIX="${LEAK_PREFIX:-${RUN_TAG}_stageB}"
export INTERVAL_SEC="${INTERVAL_SEC:-60}"
export EVAL_BS="${EVAL_BS:-2}"
python -u "${PANDA_PROJECT}/scripts/watch_sicap_stageB.py"
