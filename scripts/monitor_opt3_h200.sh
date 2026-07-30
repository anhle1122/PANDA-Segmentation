#!/usr/bin/env bash
# Monitor H200 free slots for Option 3. Prefer:
#   1) If no opt3 job: submit 2xH200 (grab partial node) with auto-resume
#   2) If running on 2 GPUs and a full 4xH200 node is free: cancel 2-GPU,
#      resubmit 4xH200 with --resume latest.pth (never scratch restart)
# Safe to run from a Cursor loop every few minutes.
set -euo pipefail
PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
cd "${PANDA_PROJECT}"
RUN_TAG="${RUN_TAG:-pseudo_r1_opt3_slidebag}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
LATEST="${CKPT_DIR}/latest.pth"
STATE_FILE="${PANDA_PROJECT}/logs/opt3_gpu_monitor.state"
mkdir -p logs

ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=== ${ts} | opt3 H200 monitor ==="

# Our jobs
mapfile -t OUR < <(squeue -u "${USER}" -n opt3_slidebag -h -o '%A %T %b %D %R' 2>/dev/null || true)
OUR_LINE="${OUR[0]:-}"
JOBID=""; STATE=""; GRES=""; NNODES=""; REASON=""
if [[ -n "${OUR_LINE}" ]]; then
  read -r JOBID STATE GRES NNODES REASON <<<"${OUR_LINE}"
fi
NGPU_REQ=0
if [[ "${GRES}" =~ gpu:h200:([0-9]+) ]]; then
  NGPU_REQ="${BASH_REMATCH[1]}"
elif [[ "${GRES}" =~ gpu:([0-9]+) ]]; then
  NGPU_REQ="${BASH_REMATCH[1]}"
fi

echo "our_job=${JOBID:-none} state=${STATE:--} gres=${GRES:--} reason=${REASON:-}"

# Per-node free H200 count (Cfg 4 minus AllocTRES gres/gpu)
declare -A FREE
for n in 095 096 097 098; do
  node="esplhpc-cp${n}"
  info="$(scontrol show node "${node}" 2>/dev/null || true)"
  if [[ -z "${info}" ]]; then
    echo "  ${node}: scontrol failed"
    FREE["${node}"]=0
    continue
  fi
  if echo "${info}" | grep -Eq 'State=.*(DRAIN|DOWN|NOT_RESPONDING)'; then
    FREE["${node}"]=0
    echo "  ${node}: drained/down → free_h200=0"
    continue
  fi
  if ! echo "${info}" | grep -q 'Gres=gpu:h200'; then
    FREE["${node}"]=0
    echo "  ${node}: not h200"
    continue
  fi
  cfg="$(echo "${info}" | sed -n 's/.*CfgTRES=[^ ]*gres\/gpu=\([0-9]*\).*/\1/p' | head -1)"
  alloc="$(echo "${info}" | sed -n 's/.*AllocTRES=[^ ]*gres\/gpu=\([0-9]*\).*/\1/p' | head -1)"
  cfg="${cfg:-4}"
  alloc="${alloc:-0}"
  free=$((cfg - alloc))
  if (( free < 0 )); then free=0; fi
  FREE["${node}"]=${free}
  echo "  ${node}: free_h200=${free} (cfg=${cfg} alloc=${alloc})"
done

max_free=0
node_with_4=""
node_with_2=""
for node in "${!FREE[@]}"; do
  f="${FREE[$node]}"
  (( f > max_free )) && max_free=$f
  if (( f >= 4 )) && [[ -z "${node_with_4}" ]]; then node_with_4="${node}"; fi
  if (( f >= 2 )) && [[ -z "${node_with_2}" ]]; then node_with_2="${node}"; fi
done

resume_args=()
if [[ -f "${LATEST}" ]]; then
  resume_args=(100 "${LATEST}")
  echo "checkpoint: ${LATEST}"
else
  echo "checkpoint: none yet (cold start OK)"
fi

submit_n() {
  local n="$1"
  local mem="64G"
  local cpus=4
  if (( n >= 4 )); then mem="400G"; cpus=16; fi
  echo "Submitting ${n}xH200 mem=${mem} cpus=${cpus} opt3 (resume=${LATEST:+yes})"
  local common=(
    --partition=preemptable --qos=part_preemptable
    --gres="gpu:h200:${n}" --mem="${mem}" --cpus-per-task="${cpus}"
    --job-name=opt3_slidebag
  )
  if ((${#resume_args[@]})); then
    sbatch "${common[@]}" scripts/slurm_train_opt3_slidebag.sh "${resume_args[@]}"
  else
    sbatch "${common[@]}" scripts/slurm_train_opt3_slidebag.sh
  fi
}

# Decision tree
if [[ -n "${JOBID}" && "${STATE}" == "RUNNING" ]]; then
  if (( NGPU_REQ <= 2 )) && [[ -n "${node_with_4}" ]]; then
    if [[ -f "${LATEST}" ]]; then
      echo "ACTION: upgrade 2→4 on ${node_with_4}; scancel ${JOBID}; resume ${LATEST}"
      scancel "${JOBID}"
      sleep 3
      submit_n 4 | tee -a "${STATE_FILE}"
    else
      echo "ACTION: 4 free but no latest.pth yet — keep 2-GPU running (avoid scratch restart)"
    fi
  else
    echo "ACTION: keep running ${NGPU_REQ}x job ${JOBID}"
  fi
elif [[ -n "${JOBID}" && "${STATE}" == "PENDING" ]]; then
  if (( NGPU_REQ >= 4 )) && [[ -n "${node_with_2}" ]] && [[ -z "${node_with_4}" ]]; then
    echo "ACTION: pending 4-GPU but only 2 free on ${node_with_2} — downsize to 2"
    scancel "${JOBID}"
    sleep 2
    submit_n 2 | tee -a "${STATE_FILE}"
  elif (( NGPU_REQ <= 2 )) && [[ -n "${node_with_4}" ]]; then
    echo "ACTION: pending 2-GPU but 4 free on ${node_with_4} — upgrade pending to 4"
    scancel "${JOBID}"
    sleep 2
    submit_n 4 | tee -a "${STATE_FILE}"
  else
    echo "ACTION: leave pending ${JOBID} (max_free=${max_free})"
  fi
else
  # No job
  if [[ -n "${node_with_4}" ]]; then
    echo "ACTION: no job + 4 free → submit 4"
    submit_n 4 | tee -a "${STATE_FILE}"
  elif [[ -n "${node_with_2}" ]]; then
    echo "ACTION: no job + 2 free → submit 2"
    submit_n 2 | tee -a "${STATE_FILE}"
  else
    echo "ACTION: no free H200 ≥2 — wait"
  fi
fi

echo "=== monitor done ==="
