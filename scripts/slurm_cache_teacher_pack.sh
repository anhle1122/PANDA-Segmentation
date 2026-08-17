#!/usr/bin/env bash
# Cache teacher pack (argmax + max-softmax) for one frozen epoch.
# Usage:
#   sbatch --gres=gpu:l40s:1 --export=ALL,RUN_TAG=...,RECIPE_VERSION=pre_lora_fix \
#     scripts/slurm_cache_teacher_pack.sh /path/to/epoch_029_cancer_Y.pth
# Pack dir: outputs/pseudo_label/teacher_${RUN_TAG}_${RECIPE_VERSION}_epNNN/
# Skip if that directory already exists. One L40S; chain with --dependency=afterany.
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
TAG="${RUN_TAG:-}"
RECIPE_VERSION="${RECIPE_VERSION:-}"
BASE="$(basename "${CKPT}" .pth)"
if [[ "${BASE}" =~ epoch_([0-9]+) ]]; then
  EP="$(printf '%03d' "$((10#${BASH_REMATCH[1]}))")"
else
  EP="${EPOCH_NUM:-unknown}"
fi
if [[ -z "${PACK_TAG:-}" ]]; then
  if [[ -z "${TAG}" || -z "${RECIPE_VERSION}" ]]; then
    echo "ERROR: set PACK_TAG or both RUN_TAG and RECIPE_VERSION (no hardcoded pack path)"
    exit 1
  fi
  PACK_TAG="${TAG}_${RECIPE_VERSION}"
fi
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/teacher_${PACK_TAG}_ep${EP}}"

if [[ -d "${OUT}" ]] && { [[ -f "${OUT}/pack_config.json" ]] || compgen -G "${OUT}/*_srcpred.h5" >/dev/null; }; then
  echo "SKIP existing pack for this tag+epoch: ${OUT}"
  echo "Never overwrite. Re-run only after moving/renaming the directory."
  exit 0
fi
mkdir -p "${OUT}"

echo "=== $(date) | teacher pack cache | ckpt=${CKPT} ==="
echo "OUT=${OUT} run_tag=${TAG} recipe=${RECIPE_VERSION} pack_tag=${PACK_TAG} ep=${EP}"
nvidia-smi -L || true

python -u scripts/cache_source_predictions.py \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT}" \
  --all-slides \
  --write-maxprob \
  --run-tag "${PACK_TAG}" \
  --lambda-slide "${LAMBDA_SLIDE:-0.3}" \
  --lambda-grade "${LAMBDA_GRADE:-0.3}" \
  --batch-size 4 \
  --num-workers 4 \
  --amp
echo "=== $(date) | teacher pack done | ${OUT} ==="
