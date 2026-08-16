#!/usr/bin/env bash
# After the ep15 L40S cache finishes: move the pack to validation-only,
# then run the referee as a pipeline test (not for training).
# Usage:
#   sbatch --dependency=afterok:5444966 scripts/slurm_relabel_ep15_validation.sh
#SBATCH --job-name=ep15_val_only
#SBATCH --partition=defq
#SBATCH -o logs/relabel_ep15_validation_%j.out
#SBATCH -e logs/relabel_ep15_validation_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/validation_only
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"

SRC="${SRC_PACK:-${PANDA_PROJECT}/outputs/pseudo_label/teacher_opt3_omar6_grouped_soft01_ep015}"
DEST="${DEST_PACK:-${PANDA_PROJECT}/outputs/pseudo_label/validation_only/teacher_ep015}"
CORR="${CORR_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/corrections_validation_ep015}"
TAU="${CONF_THRESHOLD:-0.7}"

echo "=== $(date) | relabel ep15 validation-only ==="
echo "SRC=${SRC}"
echo "DEST=${DEST}"
echo "CORR=${CORR}"

if [[ ! -d "${SRC}" && -d "${DEST}" ]]; then
  echo "SRC already moved; using DEST"
  SRC="${DEST}"
fi
if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: pack dir missing: ${SRC}"
  exit 1
fi

python - <<'PY' "${SRC}"
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "pack_config.json"
if not p.is_file():
    raise SystemExit(f"missing {p}")
status = str(json.loads(p.read_text()).get("status", "")).lower()
if status != "complete":
    raise SystemExit(f"pack not complete: status={status!r} in {p}")
print(f"pack status=complete ({p})")
PY

if [[ "${SRC}" != "${DEST}" ]]; then
  mkdir -p "$(dirname "${DEST}")"
  if [[ -e "${DEST}" ]]; then
    echo "ERROR: dest already exists: ${DEST}"
    exit 1
  fi
  mv "${SRC}" "${DEST}"
  echo "moved ${SRC} -> ${DEST}"
fi

cat > "${DEST}/VALIDATION_ONLY" <<EOF
VALIDATION_ONLY
This pack is for pipeline testing, not for training.
apply_isup_referee.py refuses this directory unless --allow-validation-only.
EOF
echo "wrote ${DEST}/VALIDATION_ONLY"

if [[ -e "${CORR}" ]]; then
  echo "ERROR: correction dest already exists: ${CORR}"
  exit 1
fi

python -u src/apply_isup_referee.py \
  --teacher-dir "${DEST}" \
  --out-dir "${CORR}" \
  --conf-threshold "${TAU}" \
  --allow-validation-only \
  --no-fail-on-g5-bias

echo "=== $(date) | validation referee done | ${CORR} ==="
