#!/usr/bin/env bash
# R2 anti-FP referee revision (SEPARATE from live R2-redo tau06_benign):
#   same teacher + tau=0.6 + include-benign
#   PLUS --protect-noncancer --protect-min-conf 0.5
#   (NC→cancer only when maxprob > 0.5; below that, classic ignore)
# Does NOT touch corrections_..._tau06_benign or the live train job.
#SBATCH --job-name=isup_ref_antifp
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_r2_antifp_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_r2_antifp_%j.err
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
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
  export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PYTHONPATH:-}"

TEACHER_DIR="${TEACHER_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/teacher_opt3_omar6_locked_locked_r2_ep014}"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/corrections_opt3_omar6_locked_r2_ep014_tau06_benign_antifp}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.6}"
SPLIT="${SPLIT:-${PANDA_PROJECT}/outputs/splits/panda_train.csv}"

echo "=== $(date) | ISUP referee R2 anti-FP | teacher=${TEACHER_DIR} ==="
PROTECT_MIN_CONF="${PROTECT_MIN_CONF:-0.5}"
echo "OUT=${OUT_DIR} tau=${CONF_THRESHOLD} include_benign=1 protect_noncancer=1 protect_min_conf=${PROTECT_MIN_CONF}"

if [[ ! -d "${TEACHER_DIR}" ]]; then
  echo "ERROR: missing teacher ${TEACHER_DIR}"
  exit 1
fi
if [[ -d "${OUT_DIR}" ]] && compgen -G "${OUT_DIR}/*" > /dev/null; then
  echo "ERROR: ${OUT_DIR} already populated — refusing overwrite"
  exit 1
fi

python -u "${PANDA_CODE_SRC}/apply_isup_referee.py" \
  --teacher-dir "${TEACHER_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --include-benign \
  --protect-noncancer \
  --protect-min-conf "${PROTECT_MIN_CONF}"

echo "=== $(date) | referee done | ${OUT_DIR} ==="
