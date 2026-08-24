#!/bin/bash
#SBATCH --job-name=visual_pixel
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --output=logs/%A_visual_pixel_%a.out
#SBATCH --error=logs/%A_visual_pixel_%a.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
FRAME_INDEX="${2:-0}"
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
TTC_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/figures/summary/time_to_contact.npz"

OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/pixels/png/pixel_y${PIXEL_Y}_x${PIXEL_X}"

cd "$PROJECT_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Per-pixel visualization"
echo "========================================"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-manual}"
echo "Task ID:   $TASK_ID"
echo "Pixel:     y=$PIXEL_Y, x=$PIXEL_X"
echo "Scene:     $SCENE_NAME"
echo "Frame:     $FRAME_INDEX"
echo "Output:    $OUTPUT_DIR"
echo "Started:   $(date)"
echo

"$PYTHON" -m analysis.visualize_timestamps \
    --mode pixel \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ttc-file "$TTC_FILE" \
    --frame "$FRAME_INDEX" \
    --pixel-y "$PIXEL_Y" \
    --pixel-x "$PIXEL_X" \
    --ttc-max-s 10 \
    --ttc-warning-s 1

echo
echo "Pixel y=$PIXEL_Y x=$PIXEL_X completed: $(date)"