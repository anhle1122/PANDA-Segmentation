#!/usr/bin/env bash
# CPU sidecar: keep a RW copy of src/scripts on outputs/_code_mirror and restore if wiped.
#SBATCH --job-name=opt3_src_keep
#SBATCH --partition=defq
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/preserve_opt3_sources_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/preserve_opt3_sources_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
export PATH="/usr/bin:/bin:${PATH:-}"
MIRROR="${PANDA_PROJECT}/outputs/_code_mirror"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
cd "${PANDA_PROJECT}"
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PATH="/usr/bin:/bin:${PATH:-}"
echo "=== $(date) | preserve Opt3 sources mirror=${MIRROR} ==="
python -u "${MIRROR}/scripts/preserve_opt3_sources.py" --interval-sec 60
echo "=== $(date) | source watcher stopped ==="
