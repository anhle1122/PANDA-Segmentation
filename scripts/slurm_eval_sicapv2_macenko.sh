#!/usr/bin/env bash
# Build Macenko(SICAP→PANDA) Test image root, then zero-shot eval on L40S.
# Usage:
#   sbatch --export=ALL,CKPT=/path/to/epoch_006_....pth \
#     scripts/slurm_eval_sicapv2_macenko.sh
#SBATCH --job-name=sicap_mac_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_macenko_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/eval_sicapv2_macenko_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED=0
export PYTHONUNBUFFERED=1

OFFICIAL="${SICAP_OFFICIAL:-/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2}"
METHOD="${STAIN_METHOD:-macenko}"
NORM_ROOT="${SICAP_NORM_ROOT:-${PANDA_PROJECT}/outputs/evaluation/sicapv2_${METHOD}_panda_root}"
CKPT="${CKPT:-${1:-}}"
# Avoid comma FOLDS in --export (sbatch truncates); default all folds in python.
FOLDS="${FOLDS:-1 2 3 4}"
FOLDS_CSV="$(echo "${FOLDS}" | tr ' ' ',')"

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: set CKPT= to a .pth"
  exit 1
fi
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; SICAPv2 eval must use L40S"
  exit 1
fi

echo "=== $(date) | build ${METHOD} root → ${NORM_ROOT} ==="
python -u "${PANDA_CODE_SRC}/build_sicapv2_macenko_root.py" \
  --official-root "${OFFICIAL}" \
  --out-root "${NORM_ROOT}" \
  --folds "${FOLDS_CSV}" \
  --method "${METHOD}"

OUT_DIR="${PANDA_PROJECT}/outputs/evaluation/sicapv2/${METHOD}_$(basename "${CKPT}" .pth)"
echo "=== $(date) | SICAPv2 ${METHOD} eval | ckpt=${CKPT} folds=${FOLDS_CSV} out=${OUT_DIR} ==="
nvidia-smi -L || true
python -u "${PANDA_CODE_SRC}/evaluate_sicapv2.py" \
  --checkpoint "${CKPT}" \
  --sicap-root "${NORM_ROOT}" \
  --folds "${FOLDS_CSV}" \
  --batch-size "${EVAL_BS:-8}" \
  --num-workers 4 \
  --amp \
  --out-dir "${OUT_DIR}"

# stamp protocol into summary
python - <<PY
import json
from pathlib import Path
p = Path("${OUT_DIR}") / "sicapv2_summary.json"
d = json.loads(p.read_text())
d["stain_norm"] = "${METHOD}"
d["stain_norm_root"] = "${NORM_ROOT}"
d["panda_reference"] = "85924446350920fb124b657160c966d7@1536,1024"
p.write_text(json.dumps(d, indent=2))
print("stamped", p)
PY
echo "=== OK $(date) | ${OUT_DIR} ==="
