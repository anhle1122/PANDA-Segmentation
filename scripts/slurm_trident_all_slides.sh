#!/usr/bin/env bash
# Trident-Otsu tissue seg + patch coords on all radboud_clean slides.
# Usage:
#   sbatch scripts/slurm_trident_all_slides.sh
# Resumes automatically (skips slides with existing contours/coords).

#SBATCH --job-name=panda_trident
#SBATCH --partition=defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data

module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg

export PYTHONPATH=src:vendor/TRIDENT
JOB_DIR=outputs/trident_all
WSI_DIR="${PANDA_DATA_ROOT}/slides"
WSI_LIST=data/trident_radboud_clean_wsi_list.csv
LOG="${JOB_DIR}/run.log"

mkdir -p "${JOB_DIR}"
python src/prepare_trident_wsi_list.py

COMMON_ARGS=(
  --wsi_dir "${WSI_DIR}"
  --job_dir "${JOB_DIR}"
  --custom_list_of_wsis "${WSI_LIST}"
  --wsi_ext .tiff
  --reader_type openslide
  --segmenter otsu
  --seg_conf_thresh 0.5
  --mag 20
  --patch_size 512
  --overlap 0
  --min_tissue_proportion 0
  --skip_errors
  --clear_dead_locks
  --max_workers 4
  --gpus -1 -1 -1 -1
)

echo "=== Trident all-slides (Otsu seg + coords) ==="
echo "  Slides: $(tail -n +2 "${WSI_LIST}" | wc -l)"
echo "  WSI_DIR=${WSI_DIR}"
echo "  JOB_DIR=${JOB_DIR}"
echo "  Started: $(date)"

{
  echo "--- task: seg ---"
  python vendor/TRIDENT/run_batch_of_slides.py --task seg "${COMMON_ARGS[@]}"
  echo "--- task: coords ---"
  python vendor/TRIDENT/run_batch_of_slides.py --task coords "${COMMON_ARGS[@]}"
} 2>&1 | tee -a "${LOG}"

echo "Finished: $(date)"
