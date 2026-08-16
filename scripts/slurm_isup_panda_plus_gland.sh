#!/usr/bin/env bash
# PANDA+ ISUP / Gleason primary+secondary on gland pixels (gt>=2), no area gate.
# Usage:
#   CHECKPOINT=... OUT=... sbatch scripts/slurm_isup_panda_plus_gland.sh
# Default: 1× L40S (H100 CUDA 802; do not take H200).
#SBATCH --job-name=isup_pp_gland
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o logs/isup_panda_plus_gland_%j.out
#SBATCH -e logs/isup_panda_plus_gland_%j.err
#SBATCH --time=04:00:00
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

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUT="${OUT:?set OUT}"
MIN_AREA_PCT="${MIN_AREA_PCT:-0.0}"

echo "=== $(date) | PANDA+ gland ISUP (thr=${MIN_AREA_PCT}, 1st+2nd, gt>=2) | ckpt=${CHECKPOINT} ==="
echo "OUT=${OUT}"
nvidia-smi -L || true
python - <<'PY'
import torch, time, os
print("GPU_NAME=", os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip())
print("torch_cuda", torch.cuda.is_available(), torch.cuda.device_count())
for i in range(1, 6):
    try:
        print("CUDA ready (try", i, ") n=", torch.cuda.device_count())
        break
    except Exception as e:
        print("CUDA wait", i, e)
        time.sleep(5)
PY

python -u src/panda_plus_gleason_mismatch.py \
  --checkpoint "${CHECKPOINT}" \
  --out "${OUT}" \
  --min-area-pct "${MIN_AREA_PCT}" \
  --pred-on-labeled-only \
  --amp \
  --batch-size 8 \
  --num-workers 4
echo "=== DONE $(date) ==="
