#!/usr/bin/env bash
# Scan official SICAPv2 patches for pen/ink (HSV + WSISegQC pen.pt).
#SBATCH --job-name=sicap_ink_scan
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_ink_scan_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_ink_scan_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p outputs/logs outputs/evaluation/sicapv2_ink
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
if [[ -f outputs/_code_mirror/scripts/hpc_use_code.sh ]]; then
  # shellcheck source=/dev/null
  source outputs/_code_mirror/scripts/hpc_use_code.sh
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if nvidia-smi -L 2>/dev/null | grep -qi H200; then
  echo "ERROR: H200 not allowed for this scan"
  exit 1
fi
echo "=== $(date) | SICAPv2 ink scan ==="
nvidia-smi -L || true
python -u "${PANDA_CODE_SRC}/scan_sicapv2_ink.py" \
  --sicap-root /common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2 \
  --with-wsisegqc \
  --device cuda \
  --gallery-n 32
echo "=== OK $(date) ==="
