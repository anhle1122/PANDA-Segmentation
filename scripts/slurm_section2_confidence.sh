#!/usr/bin/env bash
# CPU: Section 2 maxprob histograms on locked ep14 pack. Does not train.
#SBATCH --job-name=s2_maxprob
#SBATCH --partition=defq
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/section2_conf_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/section2_conf_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
export PANDA_PROJECT="${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
export PATH="/usr/bin:/bin:${PATH:-}"
cd "${PANDA_PROJECT}"
mkdir -p "${PANDA_PROJECT}/outputs/logs"
source .env.hpc 2>/dev/null || true
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
source "${PANDA_PROJECT}/outputs/_code_mirror/scripts/hpc_use_code.sh"
echo "=== $(date) | Section 2 confidence histograms ==="
python -u "${PANDA_CODE_SCRIPTS}/section2_confidence_histograms.py"
echo "=== $(date) | Section 2 done | AUTO_TRAIN=false ==="
