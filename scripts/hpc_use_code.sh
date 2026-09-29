# Prefer code mirror for training/eval; fall back to _restore.
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
if [[ -f "${PANDA_PROJECT}/outputs/_code_mirror/src/train_uni2_opt3_slidebag.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_code_mirror/src"
elif [[ -f "${PANDA_PROJECT}/outputs/_restore/src/train_uni2_opt3_slidebag.py" ]]; then
  export PANDA_CODE_SRC="${PANDA_PROJECT}/outputs/_restore/src"
else
  export PANDA_CODE_SRC="${PANDA_PROJECT}/src"
fi
export PYTHONPATH="${PANDA_CODE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
echo "PANDA_CODE_SRC=${PANDA_CODE_SRC}"
