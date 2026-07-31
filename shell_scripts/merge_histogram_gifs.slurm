#!/bin/bash
#SBATCH --job-name=merge_hist_gifs
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/merge_hist_gifs_%j.out
#SBATCH --error=logs/merge_hist_gifs_%j.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
TOF_HEIGHT="${2:?Missing ToF height}"
TOF_WIDTH="${3:?Missing ToF width}"
SCALE="${4:-0.5}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

INPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/pixels/gifs"
OUTPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/figures/gifs/histogram_grid_${TOF_HEIGHT}x${TOF_WIDTH}.gif"

cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "========================================"
echo "Merge histogram GIFs"
echo "========================================"
echo "Job ID:      ${SLURM_JOB_ID:-manual}"
echo "Host:        $(hostname)"
echo "Scene:       $SCENE_NAME"
echo "Grid:        ${TOF_HEIGHT}x${TOF_WIDTH}"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_FILE"
echo "Scale:       $SCALE"
echo "Started:     $(date)"
echo

"$PYTHON" -m analysis.merge_histogram_gifs \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_FILE" \
    --height "$TOF_HEIGHT" \
    --width "$TOF_WIDTH" \
    --scale "$SCALE"

echo
echo "Histogram GIF merge completed."
echo "Finished: $(date)"