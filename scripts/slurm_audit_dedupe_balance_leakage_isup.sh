#!/usr/bin/env bash
# Class mix before/after + leakage audit + mask-ISUP for train bags n>=5 (thr=0).
#SBATCH --job-name=dedupe_audit
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || true
mkdir -p logs
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:${PYTHONPATH:-}

echo "=== dedupe audit $(date) host=$(hostname) ==="
python -u src/analyze_dedupe_balance_leakage_isup.py \
  --min-patches 5 \
  --min-area-pct 0.0
echo "Finished: $(date)"
echo "Out: outputs/docs/slide_duplicates/audit_after_dedupe/"
