#!/usr/bin/env bash
# Sidecar: copy each finished Opt3 epoch to an immutable epoch_*.pth.
# Does not touch the live H200 train. CPU-only on defq.
#SBATCH --job-name=opt3_ckpt_keep
#SBATCH --partition=defq
#SBATCH -o logs/preserve_opt3_ckpts_%j.out
#SBATCH -e logs/preserve_opt3_ckpts_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
echo "=== $(date) | preserve Opt3 epoch snapshots ==="
python -u scripts/preserve_opt3_epoch_ckpts.py --interval-sec 15
echo "=== $(date) | preserve watcher stopped ==="
