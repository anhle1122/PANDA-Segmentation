#!/usr/bin/env bash
# Run several SICAP folds one after another on the GPUs this job already has.
# Same flags as scripts/slurm_train_sicap_ncmerge.sh (TV_*, REPLAY_*, patience).
#   sbatch --gres=gpu:l40s:2 --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088 \
#     --export=ALL,FOLDS="1 3 4",RUN_TAG_PREFIX=opt3_sicap_ft_ncmerge_replay,... \
#     scripts/slurm_train_sicap_ncmerge_foldloop.sh
#SBATCH --job-name=sicap_foldloop
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_foldloop_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/train_sicap_foldloop_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:2
#SBATCH --requeue
#SBATCH --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"

PREFIX="${RUN_TAG_PREFIX:?set RUN_TAG_PREFIX}"
FOLDS="${FOLDS:?set FOLDS}"
export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD)}"

for F in ${FOLDS}; do
  export FOLD="${F}"
  export RUN_TAG="${PREFIX}_val${F}"
  CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
  unset RESUME || true
  if [[ -f "${CKPT_DIR}/TRAINING_COMPLETE.txt" ]]; then
    echo "=== $(date) | LOOP skip fold ${F} tag=${RUN_TAG} (already complete) ==="
    continue
  fi
  if compgen -G "${CKPT_DIR}/epoch_*.pth" > /dev/null; then
    if [[ ! -f "${CKPT_DIR}/latest.pth" ]]; then
      echo "ERROR: fold ${F} has epoch snapshots but no latest.pth"
      exit 1
    fi
    export RESUME="${CKPT_DIR}/latest.pth"
    echo "=== $(date) | LOOP resume fold ${F} tag=${RUN_TAG} ==="
  else
    echo "=== $(date) | LOOP cold start fold ${F} tag=${RUN_TAG} ==="
  fi
  bash "${PANDA_PROJECT}/scripts/slurm_train_sicap_ncmerge.sh"
  echo "=== $(date) | LOOP fold ${F} finished ==="
done
echo "=== $(date) | LOOP done prefix=${PREFIX} folds=${FOLDS} ==="
