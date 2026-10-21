#!/usr/bin/env bash
# Rescore the official SICAP test for two checkpoints and redraw example tiles. L40S only.
#SBATCH --job-name=sicap_plot_check
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_plot_check_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/sicap_plot_check_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp079,esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH="${PANDA_PROJECT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED=0
export PYTHONUNBUFFERED=1
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; this check must use L40S"
  exit 1
fi

CK="${PANDA_PROJECT}/outputs/checkpoints"
EV="${PANDA_PROJECT}/outputs/evaluation/sicapv2"
python -u src/verify_sicap_pred_plots.py \
  --ckpt-a "${CK}/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth" \
  --name-a "zero-shot PANDA model" \
  --summary-a "${EV}/official_epoch_006_cancer_0.3488/sicapv2_summary.json" \
  --ckpt-b "${CK}/uni2_upernet_raw_opt3_sicap_ft_ncmerge_replay03_val2/epoch_005_cancer_0.6914.pth" \
  --name-b "SICAP fine-tuned (replay 0.3)" \
  --summary-b "${EV}/official_epoch_005_cancer_0.6914/sicapv2_summary.json" \
  --stems "${STEMS}" \
  --out-dir "${OUT_DIR:-${EV}/plot_check_zeroshot_vs_replay03}"
echo "=== OK $(date) ==="
