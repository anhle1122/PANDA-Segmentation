#!/usr/bin/env bash
# Step 4: compare pen.pt vs old HSV flags on a 20-slide sample.
#SBATCH --job-name=panda_pen_v2_cmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${PANDA_PROJECT:-/common/omarmlab/members/anh/panda_project}"
source .env.hpc 2>/dev/null || export PANDA_DATA_ROOT=/common/omarmlab/members/anh/panda_data
module load miniconda3/23.11.0-2
source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate wsi_seg
export PYTHONPATH=src

IDS=$(python -c "import pandas as pd; print(' '.join(pd.read_csv('outputs/pen_mark_detection_v2/hsv_sample_20.csv')['image_id'].tolist()))")
python src/scan_pen_marks_wsisegqc.py --slide-ids $IDS --no-npz
python src/pen_wsisegqc_preview.py --slide-ids $IDS --out-dir outputs/pen_mark_detection_v2/previews/hsv_sample
