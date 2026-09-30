#!/usr/bin/env bash
# Sequential 4-fold SICAP NC-merge CV on one 2×H200 allocation.
# Order: Val1_p20 → Val2 → Val3 → Val4 (patience 20, max 30).
# Usage:
#   sbatch --job-name=sicap_ncmerge_cv_chain --gres=gpu:h200:2 \
#     scripts/slurm_train_sicap_ncmerge_cv_chain.sh
#SBATCH --job-name=sicap_ncmerge_cv_chain
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_ncmerge_cv_chain_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_ncmerge_cv_chain_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:2
#SBATCH --requeue

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"

export EPOCHS="${EPOCHS:-30}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || true)}"

# fold:tag (canonical Val1 = patience-20 rerun tag)
CHAIN="${CHAIN:-1:opt3_sicap_ft_ncmerge_val1_p20,2:opt3_sicap_ft_ncmerge_val2,3:opt3_sicap_ft_ncmerge_val3,4:opt3_sicap_ft_ncmerge_val4}"

echo "=== $(date) | SICAP NC-merge CV CHAIN | ${CHAIN} | patience=${EARLY_STOP_PATIENCE} epochs=${EPOCHS} commit=${GIT_COMMIT} ==="
nvidia-smi -L || true

IFS=',' read -r -a STEPS <<< "${CHAIN}"
for step in "${STEPS[@]}"; do
  FOLD="${step%%:*}"
  RUN_TAG="${step#*:}"
  CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
  if compgen -G "${CKPT_DIR}/epoch_*.pth" > /dev/null; then
    echo "ERROR: refuse non-empty ${CKPT_DIR} (cold start needs empty dir)"
    exit 1
  fi
  mkdir -p "${CKPT_DIR}"
  echo "=== $(date) | CHAIN step fold=${FOLD} tag=${RUN_TAG} ==="
  export FOLD RUN_TAG
  # Reuse single-fold launcher (SBATCH headers are comments when run via bash).
  bash "${PANDA_PROJECT}/scripts/slurm_train_sicap_ncmerge.sh"
  echo "=== $(date) | CHAIN step DONE fold=${FOLD} tag=${RUN_TAG} ==="
done

echo "=== $(date) | SICAP NC-merge CV CHAIN finished all folds ==="
