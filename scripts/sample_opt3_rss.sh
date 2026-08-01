#!/usr/bin/env bash
# Sample Slurm MaxRSS for Option 3 and correlate with val/ckpt events.
# Appends CSV rows; prints OPT3_RSS_EVENT lines when training_log or latest.pth change.
set -euo pipefail
PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
RUN_TAG="${RUN_TAG:-pseudo_r1_opt3_slidebag}"
CKPT_DIR="${PANDA_PROJECT}/outputs/checkpoints/uni2_upernet_raw_${RUN_TAG}"
LATEST="${CKPT_DIR}/latest.pth"
TRAIN_LOG="${CKPT_DIR}/training_log.csv"
CSV="${PANDA_PROJECT}/logs/opt3_rss_timeline.csv"
STATE="${PANDA_PROJECT}/logs/opt3_rss_sampler.state"
JOB_NAME="${JOB_NAME:-opt3_slidebag}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
WARN_GB="${WARN_GB:-72}"

mkdir -p "${PANDA_PROJECT}/logs" "${CKPT_DIR}"
if [[ ! -f "${CSV}" ]]; then
  echo "ts,jobid,elapsed,max_rss_kb,max_rss_gb,log_rows,latest_mtime,latest_bytes,event,note" >"${CSV}"
fi

prev_rows=0
prev_latest_mtime=""
prev_latest_bytes=""
prev_max_rss_kb=0
if [[ -f "${STATE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE}" || true
fi

jobid="$(squeue -u "${USER}" -n "${JOB_NAME}" -t R -h -o '%A' 2>/dev/null | head -1 || true)"
ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
if [[ -z "${jobid}" ]]; then
  echo "${ts} | no RUNNING ${JOB_NAME}"
  exit 0
fi

elapsed="$(squeue -j "${jobid}" -h -o '%M' 2>/dev/null || echo '?')"
max_rss_kb="$(sstat -j "${jobid}.batch" --format=MaxRSS -n -P 2>/dev/null | head -1 | tr -d 'K' || true)"
if [[ -z "${max_rss_kb}" || ! "${max_rss_kb}" =~ ^[0-9]+$ ]]; then
  max_rss_kb=0
fi
# awk for float GB
max_rss_gb="$(awk -v k="${max_rss_kb}" 'BEGIN{printf "%.2f", k/1024/1024}')"

log_rows=0
if [[ -f "${TRAIN_LOG}" ]]; then
  log_rows="$(awk 'NR>1 && NF>0 {c++} END{print c+0}' "${TRAIN_LOG}")"
fi

latest_mtime=""
latest_bytes=0
if [[ -f "${LATEST}" ]]; then
  latest_mtime="$(stat -c '%Y' "${LATEST}" 2>/dev/null || true)"
  latest_bytes="$(stat -c '%s' "${LATEST}" 2>/dev/null || echo 0)"
fi

delta_kb=$((max_rss_kb - prev_max_rss_kb))
delta_gb="$(awk -v k="${delta_kb}" 'BEGIN{printf "%+.2f", k/1024/1024}')"
event="sample"
note="delta_vs_prev=${delta_gb}G"
if (( log_rows > prev_rows )); then
  event="val_log_row"
  note="training_log ${prev_rows}->${log_rows}; MaxRSSΔ=${delta_gb}G (val+metrics write window)"
elif [[ -n "${latest_mtime}" && "${latest_mtime}" != "${prev_latest_mtime}" ]]; then
  event="latest_pth_save"
  note="latest.pth mtime changed bytes=${latest_bytes}; MaxRSSΔ=${delta_gb}G"
elif [[ -n "${latest_bytes}" && "${prev_latest_bytes}" != "" && "${latest_bytes}" != "${prev_latest_bytes}" ]]; then
  event="latest_pth_save"
  note="latest.pth size ${prev_latest_bytes}->${latest_bytes}; MaxRSSΔ=${delta_gb}G"
fi

# warn if approaching limit (overrides event name for alert)
warn=0
awk -v g="${max_rss_gb}" -v w="${WARN_GB}" 'BEGIN{exit !(g+0 >= w+0)}' && warn=1 || true
if (( warn )); then
  event="rss_warn"
  note="MaxRSS ${max_rss_gb}G >= warn ${WARN_GB}G (limit 96G); ${note}"
fi

echo "${ts},${jobid},${elapsed},${max_rss_kb},${max_rss_gb},${log_rows},${latest_mtime:-},${latest_bytes},${event},${note}" >>"${CSV}"

cat >"${STATE}" <<EOF
prev_rows=${log_rows}
prev_latest_mtime='${latest_mtime}'
prev_latest_bytes='${latest_bytes}'
prev_max_rss_kb=${max_rss_kb}
EOF

echo "${ts} | job=${jobid} elapsed=${elapsed} MaxRSS=${max_rss_gb}G Δ=${delta_gb}G rows=${log_rows} latest=${latest_bytes}B event=${event}"
if [[ "${event}" != "sample" ]]; then
  echo "OPT3_RSS_EVENT event=${event} MaxRSS=${max_rss_gb}G delta=${delta_gb}G elapsed=${elapsed} note=${note}"
fi
