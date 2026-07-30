#!/bin/bash
#SBATCH --job-name=timestamp_viz
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40g
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err

set -euxo pipefail

# Usage:
#   sbatch shell_scripts/run_visualizations.slurm \
#       <scene_name> [frame] [pixel_y] [pixel_x] [make_gifs] [gif_stride]
#
# Example without GIFs:
#   sbatch shell_scripts/run_visualizations.slurm \
#       drone_flyby 10000 4 4 false 100
#
# Example with GIFs:
#   sbatch shell_scripts/run_visualizations.slurm \
#       drone_flyby 10000 4 4 true 100

SCENE_NAME="${1:?Missing scene name}"
FRAME_INDEX="${2:-0}"
PIXEL_Y="${3:-4}"
PIXEL_X="${4:-4}"
MAKE_GIFS="${5:-false}"
GIF_STRIDE="${6:-100}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

VISUALIZATION_SCRIPT="$PROJECT_DIR/analysis/visualize_timestamps.py"
INPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -f "$VISUALIZATION_SCRIPT" ]]; then
    echo "ERROR: Visualization script not found:" >&2
    echo "       $VISUALIZATION_SCRIPT" >&2
    exit 1
fi

if [[ ! -s "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found or empty:" >&2
    echo "       $INPUT_FILE" >&2
    exit 1
fi

if (( GIF_STRIDE <= 0 )); then
    echo "ERROR: GIF stride must be greater than zero." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Timestamp visualization"
echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID:-manual}"
echo "Host:         $(hostname)"
echo "Scene:        $SCENE_NAME"
echo "Input:        $INPUT_FILE"
echo "Output:       $OUTPUT_DIR"
echo "Frame/block:  $FRAME_INDEX"
echo "Pixel:        y=$PIXEL_Y, x=$PIXEL_X"
echo "Make GIFs:    $MAKE_GIFS"
echo "GIF stride:   $GIF_STRIDE"
echo "Started:      $(date)"
echo

COMMAND=(
    python
    "$VISUALIZATION_SCRIPT"
    --input "$INPUT_FILE"
    --output-dir "$OUTPUT_DIR"
    --frame "$FRAME_INDEX"
    --pixel-y "$PIXEL_Y"
    --pixel-x "$PIXEL_X"
    --ttc-window-ms 10
    --ttc-min-closing-speed 0.10
    --ttc-max-s 10
    --ttc-warning-s 1
    --skip-raw-timestamps
    --gif-stride "$GIF_STRIDE"
)

if [[ "$MAKE_GIFS" == "true" ]]; then
    COMMAND+=(--make-gifs --gif-fps 12)
elif [[ "$MAKE_GIFS" != "false" ]]; then
    echo "ERROR: make_gifs must be either true or false." >&2
    exit 1
fi

"${COMMAND[@]}"

echo
echo "Visualization completed successfully."
echo "Finished: $(date)"
echo

find "$OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -printf "%f %k KB\n" \
    | sort