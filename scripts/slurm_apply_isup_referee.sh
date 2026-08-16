#!/usr/bin/env bash
# CPU: three-way ISUP referee on a finished teacher pack.
# Usage:
#   sbatch scripts/slurm_apply_isup_referee.sh outputs/pseudo_label/teacher_opt3_omar6_grouped_soft01_ep015
#SBATCH --job-name=isup_referee
#SBATCH --partition=defq
#SBATCH -o logs/apply_isup_referee_%j.out
#SBATCH -e logs/apply_isup_referee_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/corrections
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"

TEACHER="${1:-}"
if [[ -z "${TEACHER}" || ! -d "${TEACHER}" ]]; then
  echo "ERROR: pass teacher pack dir as arg1"
  exit 1
fi
TEACHER="$(readlink -f "${TEACHER}")"
STAMP="$(date +%Y%m%d)"
NAME="$(basename "${TEACHER}")"
TAU="${CONF_THRESHOLD:-0.7}"
OUT="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/corrections/${NAME}_tau${TAU}_${STAMP}}"

echo "=== $(date) | ISUP referee | teacher=${TEACHER} ==="
echo "OUT=${OUT} tau=${TAU}"
python -u src/apply_isup_referee.py \
  --teacher-dir "${TEACHER}" \
  --out-dir "${OUT}" \
  --conf-threshold "${TAU}"
echo "=== $(date) | referee done | ${OUT} ==="
