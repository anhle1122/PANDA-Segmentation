#!/usr/bin/env bash
# Part2 ep21 confidence (divergent sample) + LSE r pre-check
#SBATCH --job-name=part2_lse
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH -o logs/part2_lse_%j.out
#SBATCH -e logs/part2_lse_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/part2_lse_precheck
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

echo "=== $(date) | Part2 confidence + LSE precheck (128G + streaming) ==="
python -u scripts/part2_confidence_and_lse_precheck.py \
  --checkpoint outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag/epoch_021_cancer_0.6356.pth \
  --sample-csv outputs/pseudo_label/opt3_ep12_ep21_divergent_gpu_sample.csv \
  --out-dir outputs/pseudo_label/part2_lse_precheck \
  --conf-thr 0.7 \
  --min-area-pct 0.05 \
  --max-patches-per-slide 64 \
  --lse-r 2 4 8 16 32 \
  --batch-size 8 \
  --num-workers 2
echo "=== DONE $(date) ==="
