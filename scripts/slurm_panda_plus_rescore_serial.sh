#!/usr/bin/env bash
# One GPU, one after another: the 16 PANDA+ leak-split scores whose separate
# jobs could not start (gpu-partition CPU cap is 32, and the 4 replay trains
# already use it). Skips any epoch whose leak-split JSON is already written.
#   sbatch scripts/slurm_panda_plus_rescore_serial.sh
#SBATCH --job-name=pp_rescore_1gpu
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/pp_rescore_1gpu_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/pp_rescore_1gpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088
#SBATCH --requeue

set -u
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
LEAK_ROOT="${PANDA_PROJECT}/outputs/docs/slide_groups/panda_plus_lookalike_gallery/leak_split"
CK="${PANDA_PROJECT}/outputs/checkpoints"

if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200"
  exit 1
fi
echo "=== $(date) | serial PANDA+ rescore on $(nvidia-smi -L | head -1) ==="

# tag  leak_prefix  epoch
ITEMS=(
  "opt3_sicap_ft_ncmerge_tv37_val1 sicapft_tv37_val1 5"
  "opt3_sicap_ft_ncmerge_tv37_val1 sicapft_tv37_val1 6"
  "opt3_sicap_ft_ncmerge_tv37_val1 sicapft_tv37_val1 7"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 3"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 4"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 6"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 7"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 12"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 13"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 14"
  "opt3_sicap_ft_ncmerge_tv37_val2 sicapft_tv37_val2 15"
  "opt3_sicap_ft_ncmerge_replay_val2 sicapft_replay_val2 2"
  "opt3_sicap_ft_ncmerge_replay_val2 sicapft_replay_val2 3"
  "opt3_sicap_ft_ncmerge_replay_val2 sicapft_replay_val2 4"
  "opt3_sicap_ft_ncmerge_replay_val2 sicapft_replay_val2 6"
  "opt3_sicap_ft_ncmerge_replay_val2 sicapft_replay_val2 10"
)

fail=0
for spec in "${ITEMS[@]}"; do
  set -- ${spec}
  tag="$1"
  pref="$2"
  ep="$3"
  ep3="$(printf '%03d' "${ep}")"
  done_json="${LEAK_ROOT}/${pref}_ep${ep3}/panda_plus_leak_split.json"
  if [[ -s "${done_json}" ]]; then
    echo "SKIP ${pref} ep${ep3} already scored"
    continue
  fi
  ck="$(ls "${CK}/uni2_upernet_raw_${tag}/epoch_${ep3}_cancer_"*.pth 2>/dev/null | head -1)"
  if [[ -z "${ck}" || ! -f "${ck}" ]]; then
    echo "MISSING ckpt ${tag} ep${ep3}"
    fail=1
    continue
  fi
  echo "=== $(date) | ${pref} ep${ep3} | ${ck} ==="
  RUN_TAG="${tag}" LEAK_PREFIX="${pref}" EPOCH="${ep}" EVAL_BS=2 \
    bash "${PANDA_PROJECT}/scripts/slurm_eval_opt3_panda_plus_r2redo.sh" "${ck}"
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "FAIL ${pref} ep${ep3} rc=${rc}"
    fail=1
  fi
done
echo "=== $(date) | serial PANDA+ rescore done fail=${fail} ==="
exit ${fail}
