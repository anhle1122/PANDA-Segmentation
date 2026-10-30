#!/usr/bin/env bash
# Install Subset1 Train TIFFs from _incoming/batch4 into the scorer folder,
# then move the download pile to _archive/incoming. CPU only. Not the login node.
#SBATCH --job-name=aggc_organize
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH -o /common/omarmlab/members/anh/panda_project/outputs/logs/organize_aggc_%j.out
#SBATCH -e /common/omarmlab/members/anh/panda_project/outputs/logs/organize_aggc_%j.err
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /common/omarmlab/members/anh/panda_project
mkdir -p outputs/logs
python3 -u scripts/organize_aggc2022_layout.py
echo "=== OK $(date) ==="
