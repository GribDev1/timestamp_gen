#!/bin/bash
#SBATCH --job-name=merge_timestamps
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/timestamp_merge_%j.out
#SBATCH --error=logs/timestamp_merge_%j.err

set -euxo pipefail

# Usage:
#   sbatch merge_timestamp_pixels.slurm \
#       <scene_name> [tof_width] [tof_height]

SCENE_NAME="${1:?Missing scene name}"
TOF_WIDTH="${2:-8}"
TOF_HEIGHT="${3:-8}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

MERGE_SCRIPT="$PROJECT_DIR/scripts/merge_timestamp_pixels.py"
PIXEL_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/pixels"
FINAL_OUTPUT="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

test -f "$MERGE_SCRIPT"
test -d "$PIXEL_DIR"

python "$MERGE_SCRIPT" \
    --pixel-dir "$PIXEL_DIR" \
    --output "$FINAL_OUTPUT" \
    --height "$TOF_HEIGHT" \
    --width "$TOF_WIDTH"

test -s "$FINAL_OUTPUT"

echo "Merged timestamp output:"
ls -lh "$FINAL_OUTPUT"