#!/usr/bin/env bash
# Modified Part2: ep21-only confidence (ISUP-mismatch + pred≠mask)
#SBATCH --job-name=part2_ep21
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH -o logs/part2_ep21_only_%j.out
#SBATCH -e logs/part2_ep21_only_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/part2_ep21_only
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

echo "=== $(date) | Part2 ep21-only confidence (128G + streaming) ==="
python -u scripts/part2_ep21_only_confidence.py \
  --checkpoint outputs/checkpoints/uni2_upernet_raw_pseudo_r1_opt3_slidebag/epoch_021_cancer_0.6356.pth \
  --diagnostic-csv outputs/pseudo_label/diagnostic_report_opt3_ep21.csv \
  --out-dir outputs/pseudo_label/part2_ep21_only \
  --conf-thr 0.7 \
  --min-area-pct 0.05 \
  --max-mismatch-slides 400 \
  --max-match-control-slides 80 \
  --max-patches-per-slide 48 \
  --batch-size 8 \
  --num-workers 2
echo "=== DONE $(date) ==="
