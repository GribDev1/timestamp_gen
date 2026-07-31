#!/bin/bash
#SBATCH --job-name=viz_hist_gif
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/viz_hist_gif_%A_%a.out
#SBATCH --error=logs/viz_hist_gif_%A_%a.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
GIF_STRIDE="${2:-100}"
TOF_HEIGHT="${3:?Missing ToF height}"
TOF_WIDTH="${4:?Missing ToF width}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array task ID}"

PIXEL_Y=$((TASK_ID / TOF_WIDTH))
PIXEL_X=$((TASK_ID % TOF_WIDTH))

if (( PIXEL_Y >= TOF_HEIGHT )); then
    echo "ERROR: Task $TASK_ID maps outside ${TOF_HEIGHT}x${TOF_WIDTH}" >&2
    exit 1
fi

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

INPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"

OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/pixels/gifs/pixel_y${PIXEL_Y}_x${PIXEL_X}"

cd "$PROJECT_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Pixel histogram GIF"
echo "========================================"
echo "Task ID:    $TASK_ID"
echo "Pixel:      y=$PIXEL_Y, x=$PIXEL_X"
echo "Scene:      $SCENE_NAME"
echo "GIF stride: $GIF_STRIDE"
echo "Output:     $OUTPUT_DIR"
echo "Started:    $(date)"
echo

"$PYTHON" -m analysis.visualize_timestamps \
    --mode histogram-gif \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --pixel-y "$PIXEL_Y" \
    --pixel-x "$PIXEL_X" \
    --gif-fps 12 \
    --gif-stride "$GIF_STRIDE"

echo
echo "Histogram GIF y=$PIXEL_Y x=$PIXEL_X completed: $(date)"