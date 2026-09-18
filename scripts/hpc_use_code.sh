# Source after PANDA_PROJECT is set.
# Prefer a tree whose trainer still has --label-source (Round 4 corrected labels).
# Order: RW mirror, live src, home backup. A /common wipe must not pick a stale trainer.
: "${PANDA_PROJECT:=/common/omarmlab/members/anh/panda_project}"
PANDA_CODE_MIRROR="${PANDA_PROJECT}/outputs/_code_mirror"
HOME_CODE_BAK="${HOME_CODE_BAK:-/home/lea14/panda_code_backup}"

_panda_trainer_ok() {
  local src_dir="$1"
  local trainer="${src_dir}/train_uni2_opt3_slidebag.py"
  [[ -f "${src_dir}/train/uni2_upernet.py" ]] || return 1
  [[ -f "${trainer}" ]] || return 1
  grep -q -- '--label-source' "${trainer}" || return 1
  grep -q -- '--corrected-dir' "${trainer}" || return 1
  return 0
}

if _panda_trainer_ok "${PANDA_CODE_MIRROR}/src"; then
  export PANDA_CODE_SRC="${PANDA_CODE_MIRROR}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_CODE_MIRROR}/scripts"
elif _panda_trainer_ok "${PANDA_PROJECT}/src"; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_PROJECT}/scripts"
  echo "WARN $(date): using live src (mirror missing --label-source)" >&2
elif _panda_trainer_ok "${HOME_CODE_BAK}/src"; then
  export PANDA_CODE_SRC="${HOME_CODE_BAK}/src"
  export PANDA_CODE_SCRIPTS="${HOME_CODE_BAK}/scripts"
  echo "WARN $(date): /common code missing flags; using ${HOME_CODE_BAK}" >&2
else
  echo "ERROR $(date): no trainer with --label-source in mirror, live, or home bak" >&2
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
  export PANDA_CODE_SCRIPTS="${PANDA_PROJECT}/scripts"
fi
export PYTHONPATH="${PANDA_CODE_SRC}:${PANDA_PROJECT}/vendor/TRIDENT:${PYTHONPATH:-}"
