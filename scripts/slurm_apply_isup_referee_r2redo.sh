#!/usr/bin/env bash
# Round-2 redo referee: same locked ep14 teacher as original Round 2, but
#   --conf-threshold 0.6
#   --include-benign  (swap-classes = benign+G3+G4+G5)
# Does NOT overwrite the original tau=0.7 G3–G5-only correction dir.
#SBATCH --job-name=isup_ref_r2redo
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_r2redo_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_r2redo_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
export PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-${PANDA_PROJECT}/data}"

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_restore/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi

REF_PY="${PANDA_CODE_SRC}/apply_isup_referee.py"
if [[ ! -f "${REF_PY}" ]]; then
  REF_PY="${PANDA_PROJECT}/outputs/_restore/src/apply_isup_referee.py"
fi
if [[ ! -f "${REF_PY}" ]]; then
  echo "ERROR: missing apply_isup_referee.py"
  exit 1
fi

TEACHER_DIR="${TEACHER_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/teacher_opt3_omar6_locked_locked_r2_ep014}"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/corrections_opt3_omar6_locked_r2_ep014_tau06_benign}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.6}"
SPLIT="${SPLIT:-${PANDA_PROJECT}/outputs/splits/panda_train.csv}"

echo "=== $(date) | ISUP referee R2-redo | teacher=${TEACHER_DIR} ==="
echo "OUT=${OUT_DIR} tau=${CONF_THRESHOLD} include_benign=1 REF_PY=${REF_PY}"
echo "PANDA_DATA_ROOT=${PANDA_DATA_ROOT}"

if [[ ! -d "${TEACHER_DIR}" ]]; then
  echo "ERROR: missing teacher ${TEACHER_DIR}"
  exit 1
fi
if [[ -d "${OUT_DIR}" ]] && compgen -G "${OUT_DIR}/*" > /dev/null; then
  echo "ERROR: ${OUT_DIR} already populated — refusing overwrite"
  exit 1
fi

python "${REF_PY}" \
  --teacher-dir "${TEACHER_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --include-benign

echo "=== $(date) | referee done | ${OUT_DIR} ==="
