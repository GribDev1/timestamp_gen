#!/bin/bash
#SBATCH --job-name=timestamp_plots
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --array=0-63%8
#SBATCH --output=logs/%A_timestamp_plots_%a.out
#SBATCH --error=logs/%A_timestamp_plots_%a.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/tvt_visual.slurm \
#       <scene_name> [tof_width] [tof_height] \
#       [start_time_ms] [end_time_ms] \
#       [y_min_ns] [y_max_ns] [marker_size]
#
# Examples:
#
# Plot the complete scene:
#   sbatch shell_scripts/tvt_visual.slurm \
#       drone_flyby
#
# Plot only 0–100 ms:
#   sbatch shell_scripts/tvt_visual.slurm \
#       drone_flyby 8 8 0 100

SCENE_NAME="${1:?Missing scene name}"
TOF_WIDTH="${2:-8}"
TOF_HEIGHT="${3:-8}"
START_TIME_MS="${4:-}"
END_TIME_MS="${5:-}"
Y_MIN_NS="${6:-0.7}"
Y_MAX_NS="${7:-26.7}"
MARKER_SIZE="${8:-0.1}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

DATASET_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"
FRAMES_DIR="$DATASET_DIR/frames"
METADATA_PATH="$DATASET_DIR/metadata.json"

FIGURES_DIR="$DATASET_DIR/figures/timestamps_vs_time"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"

PIXEL_Y=$((TASK_ID / TOF_WIDTH))
PIXEL_X=$((TASK_ID % TOF_WIDTH))

if (( PIXEL_Y >= TOF_HEIGHT )); then
    echo "ERROR: Task $TASK_ID is outside the ${TOF_HEIGHT}x${TOF_WIDTH} grid." >&2
    exit 1
fi

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -d "$FRAMES_DIR" ]]; then
    echo "ERROR: Frames directory not found:" >&2
    echo "$FRAMES_DIR" >&2
    exit 1
fi

if [[ ! -f "$METADATA_PATH" ]]; then
    echo "ERROR: Metadata file not found:" >&2
    echo "$METADATA_PATH" >&2
    exit 1
fi

FRAME_COUNT=$(
    find "$FRAMES_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'frame_*.npz' \
        | wc -l
)

if (( FRAME_COUNT == 0 )); then
    echo "ERROR: No frame NPZ files found in $FRAMES_DIR" >&2
    exit 1
fi

mkdir -p "$FIGURES_DIR"

COMMAND=(
    python
    -m analysis.visualize_timestamps
    --mode timestamps
    --input "$METADATA_PATH"
    --output-dir "$FIGURES_DIR"
    --pixel-y "$PIXEL_Y"
    --pixel-x "$PIXEL_X"
    --timestamp-marker-size "$MARKER_SIZE"
    --timestamp-y-min-ns "$Y_MIN_NS"
    --timestamp-y-max-ns "$Y_MAX_NS"
)

if [[ -n "$START_TIME_MS" ]]; then
    COMMAND+=(
        --start-time-ms "$START_TIME_MS"
    )
fi

if [[ -n "$END_TIME_MS" ]]; then
    COMMAND+=(
        --end-time-ms "$END_TIME_MS"
    )
fi

echo "=================================================="
echo "Timestamp-versus-time visualization"
echo "=================================================="
echo "Scene:          $SCENE_NAME"
echo "Task ID:        $TASK_ID"
echo "Pixel:          y=$PIXEL_Y, x=$PIXEL_X"
echo "Frame files:    $FRAME_COUNT"
echo "Dataset:        $DATASET_DIR"
echo "Output:         $FIGURES_DIR"
echo "Start time ms:  ${START_TIME_MS:-complete scene}"
echo "End time ms:    ${END_TIME_MS:-complete scene}"
echo "Host:           $(hostname)"
echo "Started:        $(date)"
echo

"${COMMAND[@]}"

EXPECTED_OUTPUT="$FIGURES_DIR/timestamps_vs_time_y${PIXEL_Y}_x${PIXEL_X}.png"

if [[ ! -s "$EXPECTED_OUTPUT" ]]; then
    echo "ERROR: Expected visualization was not created:" >&2
    echo "$EXPECTED_OUTPUT" >&2
    exit 1
fi

echo
echo "Timestamp visualization completed."
echo "Output: $EXPECTED_OUTPUT"
echo "Finished: $(date)"