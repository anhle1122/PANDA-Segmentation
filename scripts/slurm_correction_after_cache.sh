#!/usr/bin/env bash
# CPU: Step 2 validate + Step 3 referee + Step 4 G5-bias + Step 5 comparison.
# Never starts training. MODEL_ID must already be in model_registry.json.
# Usage:
#   sbatch --export=ALL,MODEL_ID=opt3_omar6_grouped_soft01_pre_lora_fix_ep029 \
#     scripts/slurm_correction_after_cache.sh
#SBATCH --job-name=isup_referee
#SBATCH --partition=defq
#SBATCH -o logs/correction_after_cache_%j.out
#SBATCH -e logs/correction_after_cache_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-}"
REGISTRY="${REGISTRY:-${PANDA_PROJECT}/outputs/pseudo_label/model_registry.json}"
if [[ -z "${MODEL_ID}" ]]; then
  echo "ERROR: MODEL_ID is required"
  exit 1
fi

echo "=== $(date) | correction after cache | model_id=${MODEL_ID} ==="
echo "AUTO_TRAIN=false"

python -u scripts/validate_correction_candidate.py \
  --model-id "${MODEL_ID}" \
  --registry "${REGISTRY}" \
  $( [[ "${ALLOW_STEP3_ANYWAY:-0}" == "1" ]] && echo --allow-step3-anyway )

read_paths() {
python - <<'PY'
import json, os
from pathlib import Path
reg = json.loads(Path(os.environ["REGISTRY"]).read_text())
entry = reg["models"][os.environ["MODEL_ID"]]
print(entry["paths"]["teacher_pack"])
print(entry["paths"]["corrections"])
print(entry.get("validation_status", ""))
PY
}
mapfile -t _PATHS < <(REGISTRY="${REGISTRY}" MODEL_ID="${MODEL_ID}" read_paths)
PACK="${_PATHS[0]}"
CORR="${_PATHS[1]}"
STATUS="${_PATHS[2]}"

if [[ ! -d "${PACK}" ]]; then
  echo "ERROR: teacher pack missing: ${PACK}"
  exit 1
fi
if [[ ! -f "${PACK}/pack_config.json" ]]; then
  echo "ERROR: pack_config.json missing (cache did not finish): ${PACK}"
  exit 1
fi

if [[ "${STATUS}" == "NOT_VALIDATED" && "${ALLOW_STEP3_ANYWAY:-0}" != "1" ]]; then
  echo "STEP3_HELD status=${STATUS} — referee not run. Review correction_comparison.md."
  python -u scripts/write_correction_comparison.py --registry "${REGISTRY}"
  exit 0
fi

export OUT_DIR="${CORR}"
python -u src/apply_isup_referee.py \
  --teacher-dir "${PACK}" \
  --out-dir "${CORR}" \
  --conf-threshold "${CONF_THRESHOLD:-0.7}" \
  --no-fail-on-g5-bias \
  $( [[ "${ALLOW_VALIDATION_ONLY:-0}" == "1" ]] && echo --allow-validation-only )

python -u scripts/write_correction_comparison.py \
  --model-id "${MODEL_ID}" \
  --registry "${REGISTRY}" \
  --balance-report "${CORR}/balance_report.json"

echo "=== $(date) | after-cache done | ${CORR} ==="
echo "Round N+1 is NOT wired. Pick from outputs/pseudo_label/correction_comparison.md"
