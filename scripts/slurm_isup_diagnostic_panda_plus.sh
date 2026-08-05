#!/usr/bin/env bash
# PANDA+ ISUP match: model paint → derive_grade vs train.csv clinical ISUP
# Usage:
#   CHECKPOINT=... OUT=... sbatch scripts/slurm_isup_diagnostic_panda_plus.sh
#SBATCH --job-name=isup_pp
#SBATCH --partition=preemptable
#SBATCH --qos=part_preemptable
#SBATCH -o logs/isup_panda_plus_%j.out
#SBATCH -e logs/isup_panda_plus_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUT="${OUT:?set OUT}"
SPLIT="${SPLIT:-outputs/splits/panda_plus_isup.csv}"
METADATA="${METADATA:-data/train.csv}"

echo "=== $(date) | PANDA+ ISUP diagnostic | ckpt=${CHECKPOINT} ==="
echo "OUT=${OUT} SPLIT=${SPLIT}"
python -u src/isup_diagnostic.py \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --metadata "${METADATA}" \
  --out "${OUT}" \
  --mode raw \
  --arch uni2_upernet \
  --batch-size 8 \
  --num-workers 4 \
  --amp \
  --amp-dtype bfloat16 \
  --min-area-pct 0.05
echo "=== DONE $(date) ==="
