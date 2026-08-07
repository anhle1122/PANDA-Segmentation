#!/usr/bin/env bash
# Shape+metadata duplicate WSI scan (Radboud clean set).
# Usage: sbatch scripts/slurm_scan_slide_duplicates.sh
# Resume: sbatch scripts/slurm_scan_slide_duplicates.sh --resume

#SBATCH --job-name=slide_dup_scan
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src:${PYTHONPATH:-}

mkdir -p outputs/docs/slide_duplicates logs
echo "=== slide duplicate scan ==="
echo "Started: $(date)  host=$(hostname)"
EXTRA=()
if [[ "${1:-}" == "--resume" ]] || [[ -f outputs/docs/slide_duplicates/fingerprints.npz ]]; then
  EXTRA+=(--resume)
  echo "Resume fingerprints if present"
fi
python src/scan_slide_duplicates.py "${EXTRA[@]}"
echo "Finished: $(date)"
echo "Artifacts under outputs/docs/slide_duplicates/"
