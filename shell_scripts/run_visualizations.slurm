#!/bin/bash
#SBATCH --job-name=visual_submit
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/visual_submit_%j.out
#SBATCH --error=logs/visual_submit_%j.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/run_visualizations.slurm \
#       <scene_name> [frame] [make_gifs] [gif_stride]
#
# Example:
#   sbatch shell_scripts/run_visualizations.slurm \
#       drone_flyby 10000 true 100

SCENE_NAME="${1:?Missing scene name}"
FRAME_INDEX="${2:-0}"
MAKE_GIFS="${3:-false}"
GIF_STRIDE="${4:-100}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

INPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"

SUMMARY_SCRIPT="$PROJECT_DIR/shell_scripts/visualization_summary.slurm"
PIXEL_SCRIPT="$PROJECT_DIR/shell_scripts/visualization_pixel.slurm"
GLOBAL_GIF_SCRIPT="$PROJECT_DIR/shell_scripts/visualization_global_gif.slurm"
HISTOGRAM_GIF_SCRIPT="$PROJECT_DIR/shell_scripts/visualization_histogram_gif.slurm"
MERGE_HISTOGRAM_GIF_SCRIPT="$PROJECT_DIR/shell_scripts/merge_histogram_gifs.slurm"


cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -s "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found or empty: $INPUT_FILE" >&2
    exit 1
fi

if [[ "$MAKE_GIFS" != "true" && "$MAKE_GIFS" != "false" ]]; then
    echo "ERROR: make_gifs must be true or false." >&2
    exit 1
fi

if ! [[ "$GIF_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: gif_stride must be a positive integer." >&2
    exit 1
fi

for script in \
    "$SUMMARY_SCRIPT" \
    "$PIXEL_SCRIPT" \
    "$GLOBAL_GIF_SCRIPT" \
    "$HISTOGRAM_GIF_SCRIPT" \
    "$MERGE_HISTOGRAM_GIF_SCRIPT"
do
    if [[ ! -f "$script" ]]; then
        echo "ERROR: Worker script not found: $script" >&2
        exit 1
    fi
done

# Read the sensor dimensions directly from the merged dataset.
read -r TOF_HEIGHT TOF_WIDTH < <(
    "$PYTHON" - "$INPUT_FILE" <<'PY'
import sys
import numpy as np

input_path = sys.argv[1]

with np.load(input_path) as data:
    shape = data["tof_depths"].shape

if len(shape) != 3:
    raise RuntimeError(
        f"Expected tof_depths shape [blocks, height, width], got {shape}"
    )

print(shape[1], shape[2])
PY
)

NUM_PIXELS=$((TOF_HEIGHT * TOF_WIDTH))
LAST_PIXEL_TASK=$((NUM_PIXELS - 1))

echo "========================================"
echo "Visualization pipeline submission"
echo "========================================"
echo "Scene:          $SCENE_NAME"
echo "Frame/block:    $FRAME_INDEX"
echo "Sensor grid:    ${TOF_HEIGHT}x${TOF_WIDTH}"
echo "Pixel tasks:    $NUM_PIXELS"
echo "Make GIFs:      $MAKE_GIFS"
echo "GIF stride:     $GIF_STRIDE"
echo

SUMMARY_SUBMISSION=$(
    sbatch --parsable \
        "$SUMMARY_SCRIPT" \
        "$SCENE_NAME" \
        "$FRAME_INDEX"
)

SUMMARY_JOB_ID="${SUMMARY_SUBMISSION%%;*}"

echo "Summary job: $SUMMARY_JOB_ID"

PIXEL_SUBMISSION=$(
    sbatch --parsable \
        --dependency="afterok:$SUMMARY_JOB_ID" \
        --array="0-${LAST_PIXEL_TASK}%8" \
        "$PIXEL_SCRIPT" \
        "$SCENE_NAME" \
        "$FRAME_INDEX" \
        "$TOF_HEIGHT" \
        "$TOF_WIDTH"
)

PIXEL_JOB_ID="${PIXEL_SUBMISSION%%;*}"

echo "Pixel PNG array: $PIXEL_JOB_ID"

if [[ "$MAKE_GIFS" == "true" ]]; then
    GLOBAL_GIF_SUBMISSION=$(
        sbatch --parsable \
            --dependency="afterok:$SUMMARY_JOB_ID" \
            --array="0-1" \
            "$GLOBAL_GIF_SCRIPT" \
            "$SCENE_NAME" \
            "$GIF_STRIDE"
    )

    GLOBAL_GIF_JOB_ID="${GLOBAL_GIF_SUBMISSION%%;*}"

    HISTOGRAM_GIF_SUBMISSION=$(
        sbatch --parsable \
            --dependency="afterok:$SUMMARY_JOB_ID" \
            --array="0-${LAST_PIXEL_TASK}%8" \
            "$HISTOGRAM_GIF_SCRIPT" \
            "$SCENE_NAME" \
            "$GIF_STRIDE" \
            "$TOF_HEIGHT" \
            "$TOF_WIDTH"
    )

    HISTOGRAM_GIF_JOB_ID="${HISTOGRAM_GIF_SUBMISSION%%;*}"

    MERGE_HISTOGRAM_GIF_SUBMISSION=$(
        sbatch --parsable \
            --dependency="afterok:$HISTOGRAM_GIF_JOB_ID" \
            "$MERGE_HISTOGRAM_GIF_SCRIPT" \
            "$SCENE_NAME" \
            "$TOF_HEIGHT" \
            "$TOF_WIDTH" \
            "0.5"
    )

    MERGE_HISTOGRAM_GIF_JOB_ID="${
        MERGE_HISTOGRAM_GIF_SUBMISSION%%;*
    }"
    echo "Global GIF array:    $GLOBAL_GIF_JOB_ID"
    echo "Histogram GIF array: $HISTOGRAM_GIF_JOB_ID"
    echo "Histogram merge job: $MERGE_HISTOGRAM_GIF_JOB_ID"
else
    echo "GIF jobs disabled."
fi

echo
echo "Submission complete."
echo
echo "Monitor with:"
echo "  squeue -u $USER"