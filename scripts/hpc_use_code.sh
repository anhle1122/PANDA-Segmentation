# Source after PANDA_PROJECT is set.
# Live src/scripts sit on NFS that is often RO (/common). outputs/ is RW.
# Always prefer the RW mirror so a /common wipe cannot break imports.
: "${PANDA_PROJECT:=/common/omarmlab/members/anh/panda_project}"
PANDA_CODE_MIRROR="${PANDA_PROJECT}/outputs/_code_mirror"
_MIRROR_CANARY="${PANDA_CODE_MIRROR}/src/train/uni2_upernet.py"
_LIVE_CANARY="${PANDA_PROJECT}/src/train/uni2_upernet.py"
if [[ -f "${_MIRROR_CANARY}" ]]; then
  export PANDA_CODE_SRC="${PANDA_CODE_MIRROR}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_CODE_MIRROR}/scripts"
elif [[ -f "${_LIVE_CANARY}" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_PROJECT}/scripts"
  echo "WARN $(date): code mirror missing; using live src" >&2
else
  echo "ERROR $(date): no uni2_upernet.py in mirror or live src" >&2
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_PROJECT}/scripts"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
