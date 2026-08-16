#!/usr/bin/env bash
# Cache teacher pack (argmax + max-softmax) for one frozen epoch.
# Usage:
#   sbatch --gres=gpu:l40s:1 scripts/slurm_cache_teacher_pack.sh \
#     outputs/checkpoints/uni2_upernet_raw_opt3_omar6_grouped_soft01/epoch_015_cancer_0.5791.pth
# Does not touch the live H200 train. Never overwrites an existing pack dir's files.
#SBATCH --job-name=teacher_pack
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o logs/cache_teacher_pack_%j.out
#SBATCH -e logs/cache_teacher_pack_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"

CKPT="${1:-}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: pass an existing checkpoint as arg1"
  exit 1
fi
CKPT="$(readlink -f "${CKPT}")"
TAG="${RUN_TAG:-opt3_omar6_grouped_soft01}"
BASE="$(basename "${CKPT}" .pth)"
if [[ "${BASE}" =~ epoch_([0-9]+) ]]; then
  EP="${BASH_REMATCH[1]}"
else
  EP="${EPOCH_NUM:-unknown}"
fi
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/teacher_${TAG}_ep${EP}}"
mkdir -p "${OUT}"

echo "=== $(date) | teacher pack cache | ckpt=${CKPT} ==="
echo "OUT=${OUT} tag=${TAG} ep=${EP}"
nvidia-smi -L || true

python -u scripts/cache_source_predictions.py \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT}" \
  --all-slides \
  --write-maxprob \
  --run-tag "${TAG}" \
  --lambda-slide "${LAMBDA_SLIDE:-0.3}" \
  --lambda-grade "${LAMBDA_GRADE:-0.3}" \
  --batch-size 4 \
  --num-workers 4 \
  --amp
echo "=== $(date) | teacher pack done | ${OUT} ==="
