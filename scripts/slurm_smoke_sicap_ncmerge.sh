#!/usr/bin/env bash
# 1-epoch SICAP nc-merge smoke on 1× L40S. Never H200. Never the live ablation tag.
#SBATCH --job-name=sicap_ncmerge_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/smoke_sicap_ncmerge_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/smoke_sicap_ncmerge_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "GATE FAIL: smoke landed on H200"
  exit 1
fi

export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
export PYTHONPATH="${PANDA_CODE_SRC}:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDNN_ENABLED=0
export OMP_NUM_THREADS=1

RUN_TAG="${RUN_TAG:-opt3_sicap_ft_ncmerge_smoke}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

INIT_CKPT="${INIT_CKPT:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth}"
UNI2_CKPT="${UNI2_CKPT:-${PANDA_PROJECT}/assets/ckpts/uni2-h/pytorch_model.bin}"
SICAP_ROOT="${SICAP_ROOT:-/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2}"

SP="${CONDA_PREFIX}/lib/python3.11/site-packages"
VENDOR_NCCL="${PANDA_PROJECT}/outputs/libs/nccl-cu11"
export LD_LIBRARY_PATH="${VENDOR_NCCL}:${SP}/nvidia/cublas/lib:${SP}/nvidia/cuda_runtime/lib:${SP}/torch/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "${VENDOR_NCCL}/libnccl.so.2" ]]; then
  export LD_PRELOAD="${VENDOR_NCCL}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

torchrun --standalone --nproc_per_node=1 \
  "${PANDA_CODE_SRC}/train_uni2_sicap_finetune.py" \
  --mode raw \
  --run-tag "${RUN_TAG}" \
  --sicap-root "${SICAP_ROOT}" \
  --fold 1 \
  --epochs 1 \
  --early-stop-patience 10 \
  --init-checkpoint "${INIT_CKPT}" \
  --slides-per-epoch 4 \
  --micro-batch-size 4 \
  --live-chunk 4 \
  --lambda-slide 0.3 \
  --lambda-grade 0.3 \
  --lambda-slide-warmup \
  --min-slide-patches 5 \
  --min-area-pct 0.0 \
  --adjacent-soft-alpha 0.1 \
  --no-include-benign-soft \
  --sicap-nc-merge \
  --decode-norm gn \
  --lora \
  --grad-checkpoint \
  --decoder-checkpoint \
  --freeze-backbone-epochs 100 \
  --amp \
  --augment \
  --num-workers 4 \
  --grad-clip 1.0 \
  --uni2-checkpoint "${UNI2_CKPT}"

SNAP="$(ls -1t "${CKPT_DIR}"/epoch_*.pth | head -1)"
OUT_DIR="${PANDA_PROJECT}/outputs/evaluation/sicapv2/$(basename "${SNAP}" .pth)"
python -u "${PANDA_CODE_SRC}/evaluate_sicapv2.py" \
  --checkpoint "${SNAP}" \
  --sicap-root "${SICAP_ROOT}" \
  --folds 1 \
  --batch-size 8 \
  --num-workers 4 \
  --amp \
  --out-dir "${OUT_DIR}"

python -u - <<PY
import csv, json, math, sys
from pathlib import Path
ckpt_dir = Path("${CKPT_DIR}")
job = "${SLURM_JOB_ID}"
outp = Path("${PANDA_PROJECT}/outputs/logs") / f"smoke_sicap_ncmerge_{job}.out"
rows = list(csv.DictReader((ckpt_dir / "training_log.csv").open()))
fail = []
if not rows:
    fail.append("empty training_log")
else:
    for k in ("train_loss", "cancer_dice", "L_pixel", "L_slide", "L_grade", "L_ce", "L_dice"):
        v = float(rows[-1][k])
        if not math.isfinite(v):
            fail.append(f"non-finite {k}={v}")
loop = float(rows[-1]["cancer_dice"]) if rows else float("nan")
bags = []
if outp.is_file():
    for line in outp.read_text(errors="replace").splitlines():
        if "BAG_LOSS" in line and "total=" in line:
            parts = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in line.split() if "=" in x}
            bags.append(float(parts["total"]))
print("BAGS", bags)
if len(bags) < 2:
    fail.append(f"need >=2 BAG_LOSS lines, got {len(bags)}")
elif not all(math.isfinite(x) for x in bags):
    fail.append("non-finite BAG_LOSS")
elif bags[-1] > bags[0]:
    fail.append(f"total loss did not decrease first={bags[0]:.4f} last={bags[-1]:.4f}")
js = json.loads(Path("${OUT_DIR}/sicapv2_summary.json").read_text())
ext = float(js["per_fold"][0]["cancer"])
delta = abs(loop - ext)
print(f"DICE_AGREE in_loop={loop:.6f} evaluate_sicapv2={ext:.6f} delta={delta:.6f}")
if delta > 0.005:
    fail.append(f"dice delta {delta:.6f} > 0.005")
trainable = False
if outp.is_file():
    trainable = any("TRAINABLE decoder=" in ln and "lora_in_optim=" in ln for ln in outp.read_text(errors="replace").splitlines())
if not trainable:
    fail.append("missing TRAINABLE decoder/grade/lora line")
path = ckpt_dir / ("GATE_FAIL.txt" if fail else "GATE_OK.txt")
path.write_text("\n".join(fail) if fail else "OK\n", encoding="utf-8")
print("GATE", "FAIL" if fail else "OK", fail)
sys.exit(1 if fail else 0)
PY
