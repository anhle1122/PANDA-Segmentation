#!/usr/bin/env bash
# Per-epoch PANDA ISUP + PANDA+ Dice + PANDA+ ISUP on 1× L40S.
# PANDA val Dice is harvested from training_log.csv (already scored each epoch).
# Usage:
#   sbatch --export=ALL,RUN_TAG=opt3_omar6_grouped_soft01,EPOCH=28,OUT_DIR=...,TRAIN_LOG=... \
#     scripts/slurm_eval_opt3_epoch.sh /path/to/epoch_028_cancer_0.xxxx.pth
# Do not request H200. Do not scancel the live train.
#SBATCH --job-name=opt3_ep_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o logs/eval_opt3_epoch_%j.out
#SBATCH -e logs/eval_opt3_epoch_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/epoch_eval outputs/evaluation
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

CKPT="${1:-}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: pass checkpoint path as arg1"
  exit 1
fi

RUN_TAG="${RUN_TAG:-unknown}"
STEM="$(basename "${CKPT}" .pth)"
if [[ -z "${EPOCH:-}" ]]; then
  EPOCH="$(python - <<PY
import re, sys
m = re.search(r"epoch_(\d+)_cancer_", "${STEM}")
print(m.group(1) if m else "")
PY
)"
fi
if [[ -z "${EPOCH}" ]]; then
  echo "ERROR: could not parse epoch from ${STEM}"
  exit 1
fi
EPOCH="$(printf '%d' "${EPOCH}")"
EP3="$(printf '%03d' "${EPOCH}")"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/epoch_eval/${RUN_TAG}/ep${EP3}}"
TRAIN_LOG="${TRAIN_LOG:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}/training_log.csv}"
SCORECARD="${SCORECARD:-${PANDA_PROJECT}/outputs/docs/opt3_this_run/epoch_external_scorecard.csv}"
mkdir -p "${OUT_DIR}"

PANDA_ISUP="${OUT_DIR}/panda_isup.csv"
PLUS_DICE="${OUT_DIR}/panda_plus_dice_labeled.csv"
PLUS_ISUP="${OUT_DIR}/panda_plus_isup.csv"
SUMMARY="${OUT_DIR}/summary.json"

echo "=== $(date) | epoch eval tag=${RUN_TAG} ep=${EPOCH} L40S | ckpt=${CKPT} ==="
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L || true

if [[ -f "${SUMMARY}" ]] && grep -q '"status": "complete"' "${SUMMARY}"; then
  echo "SKIP already complete ${SUMMARY}"
  exit 0
fi

# 1) In-domain PANDA ISUP (val split, thr=0). Never scored during train.
python -u src/isup_diagnostic.py \
  --checkpoint "${CKPT}" \
  --split "${PANDA_PROJECT}/outputs/splits/panda_val.csv" \
  --metadata "${PANDA_PROJECT}/data/train.csv" \
  --out "${PANDA_ISUP}" \
  --mode raw \
  --arch uni2_upernet \
  --batch-size 8 \
  --num-workers 4 \
  --amp \
  --amp-dtype bfloat16 \
  --min-area-pct 0.0 \
  --allow-missing-h5

# 2) PANDA+ Dice (gt>=2, classes 2–5)
python -u src/evaluate.py \
  --checkpoint "${CKPT}" \
  --arch uni2_upernet \
  --split "${PANDA_PROJECT}/outputs/panda_plus/panda_plus_patches.csv" \
  --mode raw \
  --out "${PLUS_DICE}" \
  --batch-size 4 \
  --num-workers 4 \
  --amp \
  --allow-missing-h5 \
  --mask-dir "${PANDA_PROJECT}/outputs/panda_plus/masks" \
  --mask-suffix "_pandaplus_mask.png" \
  --no-prefer-h5-masks \
  --panda-plus-eval

# 3) PANDA+ gland ISUP (thr=0, pred on labeled pixels) — same protocol as ep7/ep15
python -u src/panda_plus_gleason_mismatch.py \
  --checkpoint "${CKPT}" \
  --out "${PLUS_ISUP}" \
  --min-area-pct 0.0 \
  --pred-on-labeled-only \
  --amp \
  --batch-size 8 \
  --num-workers 4

python -u scripts/summarize_epoch_eval.py \
  --tag "${RUN_TAG}" \
  --ckpt "${CKPT}" \
  --train-log "${TRAIN_LOG}" \
  --panda-isup-summary "${OUT_DIR}/panda_isup_summary.json" \
  --panda-plus-labeled "${PLUS_DICE}" \
  --panda-plus-isup-summary "${OUT_DIR}/panda_plus_isup_summary.json" \
  --out-json "${SUMMARY}" \
  --scorecard "${SCORECARD}" \
  --job-id "${SLURM_JOB_ID:-}"

echo "=== DONE $(date) ==="
echo "Summary: ${SUMMARY}"
