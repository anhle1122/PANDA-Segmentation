#!/usr/bin/env bash
# Per-epoch PANDA+ for Round 4: leak-split Dice (all 48 / leaked 18 / clean 30)
# plus gland ISUP. Same protocol as Round 3. Never H200. Exclude cp087 / cp088.
# Usage:
#   sbatch --export=ALL,RUN_TAG=opt3_omar6_round4_ep6ref,EPOCH=1 \
#     scripts/slurm_eval_opt3_panda_plus_r4.sh /path/to/epoch_001_cancer_0.xxxx.pth
# Does not scancel the H200 train.
#SBATCH --job-name=pp_r4_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/eval_opt3_pp_r4_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/eval_opt3_pp_r4_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp087,esplhpc-cp088

set -u
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
mkdir -p logs outputs/pseudo_label/epoch_eval outputs/evaluation "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

MIRROR="${PANDA_PROJECT}/outputs/_code_mirror"
RESTORE="${PANDA_PROJECT}/outputs/_restore"
NEED=(evaluate.py evaluate_panda_plus_leak_split.py panda_plus_gleason_mismatch.py patch_utils.py isup_diagnostic.py train/__init__.py train/baseline_dataset.py train/model.py train/uni2_upernet.py train/lora_vit.py)
SEARCH=("${MIRROR}/src" "${RESTORE}/src" "${PANDA_PROJECT}/src")
SRC="${MIRROR}/src"
mkdir -p "${SRC}" "${RESTORE}/src"
usable() { [[ -f "$1" && -s "$1" ]]; }
# Self-heal: copy any missing/empty required file from another tree so one
# wipe cannot split the set across directories again. Also pin a copy in
# _restore/src so the next wipe still has a backup.
for f in "${NEED[@]}"; do
  mkdir -p "${SRC}/$(dirname "${f}")" "${RESTORE}/src/$(dirname "${f}")"
  if usable "${SRC}/${f}"; then
    if ! usable "${RESTORE}/src/${f}"; then
      cp -f "${SRC}/${f}" "${RESTORE}/src/${f}"
      echo "PINNED ${f} -> ${RESTORE}/src"
    fi
    continue
  fi
  recovered=""
  for cand in "${SEARCH[@]}"; do
    if usable "${cand}/${f}"; then
      cp -f "${cand}/${f}" "${SRC}/${f}"
      echo "RECOVERED ${f} from ${cand} -> ${SRC}"
      recovered=1
      break
    fi
  done
  if [[ -z "${recovered}" ]]; then
    echo "ERROR: missing ${f}"
    echo "searched:"
    for cand in "${SEARCH[@]}"; do
      echo "  ${cand}/${f}  $( usable "${cand}/${f}" && echo OK || echo MISSING )"
    done
    exit 1
  fi
  if ! usable "${RESTORE}/src/${f}"; then
    cp -f "${SRC}/${f}" "${RESTORE}/src/${f}"
    echo "PINNED ${f} -> ${RESTORE}/src"
  fi
done
for f in "${NEED[@]}"; do
  if ! usable "${SRC}/${f}"; then
    echo "ERROR: ${SRC}/${f} still missing after recover"
    exit 1
  fi
done
echo "SRC=${SRC} files=${NEED[*]}"
SCRIPTS=""
for cand in "${MIRROR}/scripts" "${PANDA_PROJECT}/scripts"; do
  if [[ -f "${cand}/summarize_epoch_eval.py" ]]; then
    SCRIPTS="${cand}"
    break
  fi
done
if [[ -z "${SCRIPTS}" ]]; then
  echo "ERROR: missing summarize_epoch_eval.py"
  exit 1
fi
export PYTHONPATH="${SRC}:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TORCH_CUDNN_ENABLED="${TORCH_CUDNN_ENABLED:-0}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EVAL_BS="${EVAL_BS:-8}"
TAG="${RUN_TAG:-opt3_omar6_round4_ep6ref}"
CKPT="${1:-}"
LEAK_ROOT="${PANDA_PROJECT}/outputs/docs/slide_groups/panda_plus_lookalike_gallery/leak_split"
DUP_CSV="${PANDA_PROJECT}/outputs/docs/slide_groups/panda_plus_lookalike_gallery/panda_plus_confirmed_duplicates.csv"

echo "=== $(date) | PANDA+ Round 4 leak-split | SRC=${SRC} tag=${TAG} ckpt=${CKPT} ==="
nvidia-smi -L || true
if nvidia-smi -L 2>/dev/null | grep -qi 'H200'; then
  echo "ERROR: landed on H200; PANDA+ eval must use L40S/H100/A100"
  exit 1
fi
python - <<'PY'
import torch, sys
print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch cannot see CUDA; refusing CPU fallback")
print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: pass checkpoint path as arg1"
  exit 1
fi
STEM="$(basename "${CKPT}" .pth)"
if [[ -z "${EPOCH:-}" ]]; then
  EPOCH="$(python - <<PY
import re
m = re.search(r"epoch_(\d+)_cancer_", "${STEM}")
print(m.group(1) if m else "")
PY
)"
fi
if [[ -z "${EPOCH}" ]]; then
  echo "ERROR: could not parse epoch from ${STEM}"
  exit 1
fi
EPOCH="$(printf '%d' "${EPOCH}")"
EP3="$(printf '%03d' "${EPOCH}")"
OUT_DIR="${OUT_DIR:-${PANDA_PROJECT}/outputs/pseudo_label/epoch_eval/${TAG}/ep${EP3}}"
LEAK_DIR="${LEAK_ROOT}/round4_ep${EP3}"
PLUS_DICE="${OUT_DIR}/panda_plus_dice_labeled.csv"
PLUS_ISUP="${OUT_DIR}/panda_plus_isup.csv"
TRAIN_LOG="${TRAIN_LOG:-${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${TAG}/training_log.csv}"
SCORECARD="${PANDA_PROJECT}/outputs/docs/opt3_this_run/epoch_external_scorecard.csv"
SUMMARY="${OUT_DIR}/summary.json"
mkdir -p "${OUT_DIR}" "${LEAK_DIR}"

if [[ -s "${LEAK_DIR}/panda_plus_leak_split.json" && -s "${OUT_DIR}/panda_plus_clean30.json" && -s "${PLUS_ISUP}" ]]; then
  echo "SKIP ${TAG} ep${EP3} already scored"
else
  if [[ ! -s "${LEAK_DIR}/panda_plus_leak_split.json" || ! -s "${PLUS_DICE}" ]]; then
    echo "=== $(date) | leak-split dice ${TAG} ep=${EPOCH} | ${CKPT} ==="
    python -u "${SRC}/evaluate_panda_plus_leak_split.py" \
      --checkpoint "${CKPT}" \
      --out-dir "${LEAK_DIR}" \
      --dice-csv "${PLUS_DICE}" \
      --batch-size "${EVAL_BS}" \
      --num-workers 4 \
      --amp
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "FAIL leak-split ${TAG} ep${EP3} rc=${rc}"
      exit $rc
    fi
  else
    echo "SKIP leak-split ${TAG} ep${EP3} already scored"
  fi

  if [[ ! -s "${PLUS_ISUP}" || ! -s "${OUT_DIR}/panda_plus_isup_summary.json" ]]; then
    echo "=== $(date) | ISUP ${TAG} ep=${EPOCH} ==="
    python -u "${SRC}/panda_plus_gleason_mismatch.py" \
      --checkpoint "${CKPT}" \
      --out "${PLUS_ISUP}" \
      --min-area-pct 0.0 \
      --pred-on-labeled-only \
      --amp \
      --batch-size "${EVAL_BS}" \
      --num-workers 4
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "FAIL isup ${TAG} ep${EP3} rc=${rc}"
      exit $rc
    fi
  else
    echo "SKIP isup ${TAG} ep${EP3} already scored"
  fi

  python - <<PY
import csv, json
from pathlib import Path
dup = Path("${DUP_CSV}")
leaked = {row["plus_id"] for row in csv.DictReader(dup.open())}
isup_path = Path("${PLUS_ISUP}")
rows = list(csv.DictReader(isup_path.open())) if isup_path.is_file() else []
def rate(subset):
    if not subset:
        return None
    hits = sum(1 for r in subset if str(r.get("isup_match","")).lower() in {"true","1","yes"})
    return hits / len(subset)
all_r = rate(rows)
clean_r = rate([r for r in rows if r["slide_id"] not in leaked])
leaked_r = rate([r for r in rows if r["slide_id"] in leaked])
leak_json = Path("${LEAK_DIR}/panda_plus_leak_split.json")
payload = json.loads(leak_json.read_text()) if leak_json.is_file() else {}
payload["panda_plus_isup_match_all48"] = all_r
payload["panda_plus_isup_match_clean30"] = clean_r
payload["panda_plus_isup_match_leaked18"] = leaked_r
payload["n_isup_slides"] = len(rows)
if leak_json.is_file():
    leak_json.write_text(json.dumps(payload, indent=2) + "\n")
g = {x["group"]: x for x in payload.get("groups", [])}
out = Path("${OUT_DIR}/panda_plus_clean30.json")
out.write_text(json.dumps({
    "tag": "${TAG}",
    "epoch": ${EPOCH},
    "cancer_dice_all48": payload.get("headline_cancer_dice"),
    "cancer_dice_clean30": payload.get("clean_only_cancer_dice"),
    "cancer_dice_leaked18": (g.get("leaked") or {}).get("cancer_dice"),
    "headline_minus_clean": payload.get("headline_minus_clean"),
    "isup_all48": all_r,
    "isup_clean30": clean_r,
    "isup_leaked18": leaked_r,
}, indent=2) + "\n")
print("CLEAN30", json.dumps({
    "ep": ${EPOCH},
    "dice_all": payload.get("headline_cancer_dice"),
    "dice_clean30": payload.get("clean_only_cancer_dice"),
    "isup_all": all_r,
    "isup_clean30": clean_r,
}))
PY

  python -u "${SCRIPTS}/summarize_epoch_eval.py" \
    --tag "${TAG}" \
    --ckpt "${CKPT}" \
    --train-log "${TRAIN_LOG}" \
    --panda-plus-labeled "${PLUS_DICE}" \
    --panda-plus-isup-summary "${OUT_DIR}/panda_plus_isup_summary.json" \
    --out-json "${OUT_DIR}/panda_plus_only.json" \
    --scorecard "${SCORECARD}" \
    --job-id "${SLURM_JOB_ID:-batch}"
  python - <<PY
import json
from pathlib import Path
only = Path("${OUT_DIR}/panda_plus_only.json")
summary = Path("${SUMMARY}")
payload = json.loads(only.read_text()) if only.is_file() else {}
payload["status"] = "complete"
payload["protocol"] = "panda_plus_leak_split_clean30"
summary.write_text(json.dumps(payload, indent=2) + "\n")
PY
fi

python - <<PY
import csv, json
from pathlib import Path
root = Path("${LEAK_ROOT}")
rows = []
for p in sorted(root.glob("round4_ep*/panda_plus_leak_split.json")):
    d = json.loads(p.read_text())
    g = {x["group"]: x for x in d.get("groups", [])}
    ep = int(p.parent.name.replace("round4_ep", ""))
    rows.append({
        "epoch": ep,
        "dice_all48": d.get("headline_cancer_dice"),
        "dice_clean30": d.get("clean_only_cancer_dice"),
        "dice_leaked18": g.get("leaked", {}).get("cancer_dice"),
        "headline_minus_clean": d.get("headline_minus_clean"),
        "isup_all48": d.get("panda_plus_isup_match_all48"),
        "isup_clean30": d.get("panda_plus_isup_match_clean30"),
        "isup_leaked18": d.get("panda_plus_isup_match_leaked18"),
    })
out = root / "round4_all_epochs_clean30.csv"
if rows:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out, "n", len(rows))
PY

echo "=== OK ${TAG} ep${EP3} $(date) ==="
