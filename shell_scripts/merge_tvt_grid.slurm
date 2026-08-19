#!/bin/bash
#SBATCH --job-name=merge_tvt_grid
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/merge_tvt_grid_%j.out
#SBATCH --error=logs/merge_tvt_grid_%j.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/merge_tvt_grid.slurm \
#       <scene_name> [TOF_WIDTH] [TOF_HEIGHT]

SCENE_NAME="${1:?Missing scene name}"
TOF_WIDTH="${2:-8}"
TOF_HEIGHT="${3:-8}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

INPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/timestamps_vs_time"
OUTPUT_PATH="$PROJECT_DIR/outputs/$SCENE_NAME/figures/timestamps_vs_time_grid_${TOF_HEIGHT}x${TOF_WIDTH}.png"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

python analysis/merge_timestamp_plot_grid.py \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_PATH" \
    --tof-width "$TOF_WIDTH" \
    --tof-height "$TOF_HEIGHT" \
    --label-tiles

echo "Finished: $(date)"
echo "Output: $OUTPUT_PATH"