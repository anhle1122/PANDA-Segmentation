#!/usr/bin/env bash
# Bulk-download PANDA competition zip on HPC, unzip, link Radboud matched pairs.
# NEVER write to /tmp — only /common/omarmlab/members/anh/panda_data/downloads/
#
# Usage:
#   bash scripts/hpc_bulk_download_panda.sh
#   bash scripts/hpc_bulk_download_panda.sh unzip-only
#   bash scripts/hpc_bulk_download_panda.sh link-only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT"

if [[ -f "$PROJECT/.env.hpc" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT/.env.hpc"
fi

DATA_ROOT="${PANDA_DATA_ROOT:-/common/omarmlab/members/anh/panda_data}"
DOWNLOADS="$DATA_ROOT/downloads"
EXTRACT="$DOWNLOADS/panda_bulk"
ZIP="$DOWNLOADS/prostate-cancer-grade-assessment.zip"
COMP="prostate-cancer-grade-assessment"
MODE="${1:-all}"

if [[ ! -f "$HOME/.kaggle/access_token" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "ERROR: Missing Kaggle credentials in ~/.kaggle/"
    exit 1
fi

echo "=== PANDA bulk download (Radboud pairs via symlinks) ==="
echo "  Data root: $DATA_ROOT"
echo "  Zip:       $ZIP"
echo "  Extract:   $EXTRACT"
df -h "$DATA_ROOT" | tail -1

download_zip() {
    mkdir -p "$DOWNLOADS"
    if [[ -f "$ZIP" ]] && [[ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -gt 1000000 ]]; then
        echo "  Zip already present ($(du -h "$ZIP" | cut -f1)). Skipping download."
        return 0
    fi
    echo "  Downloading full competition zip (~412 GB)..."
    kaggle competitions download -c "$COMP" -p "$DOWNLOADS"
    echo "  Download complete: $(du -h "$ZIP" | cut -f1)"
}

unzip_bulk() {
    if find "$EXTRACT" -maxdepth 2 -type d -name train_images 2>/dev/null | grep -q .; then
        echo "  train_images/ already present. Skipping unzip."
        return 0
    fi
    if [[ ! -f "$ZIP" ]]; then
        echo "ERROR: Missing $ZIP"
        exit 1
    fi
    mkdir -p "$EXTRACT"
    echo "  Unzipping (may take 1-3 hours)..."
    unzip -o "$ZIP" -d "$EXTRACT"
    echo "  Unzip complete."
}

link_radboud() {
    echo "  Linking Radboud matched slide+mask pairs..."
    python src/link_radboud_bulk.py --extract-dir "$EXTRACT" --force
}

case "$MODE" in
    all)
        download_zip
        unzip_bulk
        link_radboud
        ;;
    download-only)
        download_zip
        ;;
    unzip-only)
        unzip_bulk
        ;;
    link-only)
        link_radboud
        ;;
    *)
        echo "Unknown mode: $MODE  (use: all | download-only | unzip-only | link-only)"
        exit 1
        ;;
esac

echo ""
echo "=== Bulk setup complete ($MODE) ==="
echo "  Next: python src/clean_dataset.py --local-only --resume --sleep 0"
