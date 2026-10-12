#!/usr/bin/env bash
# Closing / hole-fill diagnostic on SICAPv2 (Val fold + official), 1x L40S.
# Original eval summaries are NOT touched; output -> outputs/evaluation/postproc_sweep/.
#   sbatch scripts/slurm_postproc_sweep.sh
#SBATCH --job-name=postproc_sweep
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/postproc_sweep_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/postproc_sweep_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 TORCH_CUDNN_ENABLED=0 PYTHONUNBUFFERED=1
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then echo "ERROR: H200"; exit 1; fi

CK="${PANDA_PROJECT}/outputs/checkpoints"
OUT="${PANDA_PROJECT}/outputs/evaluation/postproc_sweep"
R3="${CK}/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth"
PY="python ${PANDA_PROJECT}/src/postproc_sweep_sicap.py --amp"

run() { # label partition fold (--checkpoint X | --run-dir X)
  local label="$1" part="$2" fold="$3"; shift 3
  local f="${OUT}/${label}_${part}_f${fold}.json"
  [[ -f "${f}" ]] && { echo "skip ${f}"; return; }
  ${PY} --label "${label}" --partition "${part}" --fold "${fold}" --out-json "${f}" "$@"
}

# Order: fast, most informative first.
run r3ep6 official 1 --checkpoint "${R3}"
run heads_v1 val 1 --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val1_p20"
run heads_v1 official 1 --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val1_p20"
run seg_v1 val 1 --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val1_pixelonly"
for n in 2 3 4; do
  run heads_v${n} val ${n} --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val${n}"
  run seg_v${n} val ${n} --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val${n}_pixelonly"
done
for n in 1 2 3 4; do run r3ep6 val ${n} --checkpoint "${R3}"; done
for n in 2 3 4; do
  run heads_v${n} official ${n} --run-dir "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_val${n}"
done
echo "=== $(date) | postproc sweep done ==="
