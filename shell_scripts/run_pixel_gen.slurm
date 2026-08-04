#!/bin/bash
#SBATCH --job-name=timestamp_pixel
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --output=logs/timestamp_pixel_%A_%a.out
#SBATCH --error=logs/timestamp_pixel_%A_%a.err

set -euxo pipefail

# Usage:
#   sbatch --array=0-63%8 \
#       shell_scripts/run_pixel_gen.slurm \
#       <scene_name> [sensor_name] [render_fps] [tof_width] [tof_height]

SCENE_NAME="${1:?Missing scene name}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"
TOF_WIDTH="${4:-8}"
TOF_HEIGHT="${5:-8}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

TIMESTAMP_SCRIPT="$PROJECT_DIR/timestamp_gen.py"
RENDER_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"
PIXEL_OUTPUT_DIR="$OUTPUT_DIR/pixels"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

PIXEL_Y=$((TASK_ID / TOF_WIDTH))
PIXEL_X=$((TASK_ID % TOF_WIDTH))

if (( PIXEL_Y >= TOF_HEIGHT )); then
    echo "ERROR: Task $TASK_ID maps outside ${TOF_HEIGHT}x${TOF_WIDTH} grid." >&2
    exit 1
fi

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$PIXEL_OUTPUT_DIR"

echo "Scene: $SCENE_NAME"
echo "Sensor: $SENSOR_NAME"
echo "Task: $TASK_ID"
echo "Pixel: y=$PIXEL_Y, x=$PIXEL_X"
echo "Host: $(hostname)"
echo "Start: $(date)"

python "$TIMESTAMP_SCRIPT" \
    --sensor "$SENSOR_NAME" \
    --render-dir "$RENDER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --render-fps "$RENDER_FPS" \
    --pixel-y "$PIXEL_Y" \
    --pixel-x "$PIXEL_X"

PIXEL_FILE="$PIXEL_OUTPUT_DIR/pixel_y${PIXEL_Y}_x${PIXEL_X}.npz"

if [[ ! -s "$PIXEL_FILE" ]]; then
    echo "ERROR: Missing pixel output: $PIXEL_FILE" >&2
    exit 1
fi

echo "Pixel generation completed."
echo "Output: $PIXEL_FILE"
echo "Finished: $(date)"