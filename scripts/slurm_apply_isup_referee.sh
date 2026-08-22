#!/usr/bin/env bash
# CPU: three-way ISUP referee on a finished teacher pack (τ=0.7).
# Never starts training. After referee, compares against original wmfix Rules 1-3.
# Usage:
#   sbatch --export=ALL,CONF_THRESHOLD=0.7 \
#     outputs/_code_mirror/scripts/slurm_apply_isup_referee.sh \
#     outputs/pseudo_label/teacher_opt3_omar6_locked_locked_r2_ep014
#SBATCH --job-name=isup_referee
#SBATCH --partition=defq
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/apply_isup_referee_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/corrections "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"

TEACHER="${1:-}"
if [[ -z "${TEACHER}" || ! -d "${TEACHER}" ]]; then
  echo "ERROR: pass teacher pack dir as arg1"
  exit 1
fi
TEACHER="$(readlink -f "${TEACHER}")"
NAME="$(basename "${TEACHER}")"
TAU="${CONF_THRESHOLD:-0.7}"
# teacher_<pack_tag>_epNNN -> corrections_<pack_tag>_epNNN (recipe lives in pack_tag)
if [[ "${NAME}" == teacher_* ]]; then
  CORR_NAME="corrections_${NAME#teacher_}"
else
  CORR_NAME="corrections_${NAME}"
fi
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/${CORR_NAME}}"
if [[ -e "${OUT}" ]]; then
  echo "SKIP existing correction dir (never overwrite): ${OUT}"
  exit 0
fi

echo "=== $(date) | ISUP referee | teacher=${TEACHER} ==="
echo "OUT=${OUT} tau=${TAU}"
EXTRA=()
if [[ "${ALLOW_VALIDATION_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--allow-validation-only)
fi
python -u "${PANDA_CODE_SRC}/apply_isup_referee.py" \
  --teacher-dir "${TEACHER}" \
  --out-dir "${OUT}" \
  --split "${PANDA_PROJECT}/outputs/splits/panda_train.csv" \
  --clinical-csv "${PANDA_PROJECT}/data/train.csv" \
  --conf-threshold "${TAU}" \
  --no-fail-on-g5-bias \
  "${EXTRA[@]}"
echo "=== $(date) | referee done | ${OUT} ==="

python -u "${PANDA_CODE_SCRIPTS}/compare_referee_vs_wmfix.py" \
  --referee-dir "${OUT}"
echo "=== $(date) | wmfix compare done | AUTO_TRAIN=false ==="
