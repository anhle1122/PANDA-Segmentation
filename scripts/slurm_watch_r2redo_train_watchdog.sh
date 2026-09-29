#!/usr/bin/env bash
# Self-heal watchdog for Round-2-redo train (+ keep PANDA+ watcher alive).
# CPU-only. Never scancels a healthy train. Never touches Round 4.
#SBATCH --job-name=opt3_r2redo_watchdog
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_train_watchdog_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/watch_r2redo_train_watchdog_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs" "${PANDA_PROJECT}/outputs/pseudo_label"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export INTERVAL_SEC="${INTERVAL_SEC:-90}"
# Pin a copy so a scripts/ wipe can be recovered from mirror next boot
mkdir -p "${PANDA_PROJECT}/outputs/_code_mirror/scripts"
cp -f "${PANDA_PROJECT}/scripts/watch_r2redo_train_watchdog.py" \
  "${PANDA_PROJECT}/outputs/_code_mirror/scripts/watch_r2redo_train_watchdog.py" || true
cp -f "${PANDA_PROJECT}/scripts/slurm_watch_r2redo_train_watchdog.sh" \
  "${PANDA_PROJECT}/outputs/_code_mirror/scripts/slurm_watch_r2redo_train_watchdog.sh" || true
python -u "${PANDA_PROJECT}/scripts/watch_r2redo_train_watchdog.py"
