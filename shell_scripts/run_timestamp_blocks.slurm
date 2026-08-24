#!/bin/bash
#SBATCH --job-name=timestamp_blocks
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --output=logs/%A_timestamp_blocks_%a.out
#SBATCH --error=logs/%A_timestamp_blocks_%a.err

set -euo pipefail

# Each array task generates one range of complete timestamp blocks.
#
# Usage:
#   sbatch --array=0-38%8 \
#       shell_scripts/run_timestamp_blocks.slurm \
#       <scene_name> [sensor_name] [render_fps] \
#       [blocks_per_task] [block_size]
#
# Example:
#   sbatch --array=0-38%8 \
#       shell_scripts/run_timestamp_blocks.slurm \
#       drone_flyby vl53l8ch 240 1000 256
#
# Block ranges are zero-based and end-exclusive:
#   task 0 -> [0, 1000)
#   task 1 -> [1000, 2000)
#   ...
#
# The last task is automatically clipped to the scene's total block count.

SCENE_NAME="${1:?Missing scene name}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"
BLOCKS_PER_TASK="${4:-1000}"
BLOCK_SIZE="${5:-}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

TIMESTAMP_SCRIPT="$PROJECT_DIR/timestamp_gen.py"
RENDER_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"
DEPTH_DIR="$RENDER_DIR/depths"
NORMAL_DIR="$RENDER_DIR/normals"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"
FRAMES_DIR="$OUTPUT_DIR/frames"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array task ID}"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -f "$TIMESTAMP_SCRIPT" ]]; then
    echo "ERROR: Timestamp generator not found: $TIMESTAMP_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DEPTH_DIR" ]]; then
    echo "ERROR: Depth directory not found: $DEPTH_DIR" >&2
    exit 1
fi

if [[ ! -d "$NORMAL_DIR" ]]; then
    echo "ERROR: Normal directory not found: $NORMAL_DIR" >&2
    exit 1
fi

if ! [[ "$BLOCKS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: blocks_per_task must be a positive integer." >&2
    exit 1
fi

if ! [[ "$RENDER_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: render_fps must be a positive number." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$FRAMES_DIR"

TOTAL_BLOCKS=$(
    python - "$SENSOR_NAME" "$DEPTH_DIR" "$RENDER_FPS" "$BLOCK_SIZE" <<'PY'
import math
import sys
from pathlib import Path

from configs.sensor_presets import get_sensor_preset

sensor_name = sys.argv[1]
depth_dir = Path(sys.argv[2])
render_fps = float(sys.argv[3])
block_size_arg = sys.argv[4]

if render_fps <= 0:
    raise SystemExit("render_fps must be positive")

sensor = get_sensor_preset(sensor_name)

block_size = (
    int(block_size_arg)
    if block_size_arg
    else sensor.block_size_L
)

if block_size <= 0:
    raise SystemExit("block size must be positive")

depth_files = sorted(depth_dir.glob("*.exr"))

if len(depth_files) < 2:
    raise SystemExit(
        "At least two depth EXR files are required; "
        f"found {len(depth_files)}"
    )

render_intervals = len(depth_files) - 1
scene_duration_s = render_intervals / render_fps

block_duration_s = (
    block_size / sensor.laser_rate_hz
)

total_blocks = int(
    math.floor(scene_duration_s / block_duration_s)
)

print(total_blocks)
PY
)

if ! [[ "$TOTAL_BLOCKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Invalid total block count: $TOTAL_BLOCKS" >&2
    exit 1
fi

START_BLOCK=$((TASK_ID * BLOCKS_PER_TASK))
END_BLOCK=$((START_BLOCK + BLOCKS_PER_TASK))

if (( START_BLOCK >= TOTAL_BLOCKS )); then
    echo "Task $TASK_ID has no assigned blocks."
    echo "Start block:  $START_BLOCK"
    echo "Total blocks: $TOTAL_BLOCKS"
    exit 0
fi

if (( END_BLOCK > TOTAL_BLOCKS )); then
    END_BLOCK="$TOTAL_BLOCKS"
fi

BLOCK_COUNT=$((END_BLOCK - START_BLOCK))

FIRST_FRAME=$((START_BLOCK + 1))
LAST_FRAME="$END_BLOCK"

FIRST_FRAME_PATH=$(
    printf "%s/frame_%06d.npz" \
        "$FRAMES_DIR" \
        "$FIRST_FRAME"
)

LAST_FRAME_PATH=$(
    printf "%s/frame_%06d.npz" \
        "$FRAMES_DIR" \
        "$LAST_FRAME"
)

echo "=================================================="
echo "Full timestamp block generation"
echo "=================================================="
echo "Scene:             $SCENE_NAME"
echo "Sensor:            $SENSOR_NAME"
echo "Render FPS:        $RENDER_FPS"
echo "Task ID:           $TASK_ID"
echo "Blocks per task:   $BLOCKS_PER_TASK"
echo "Total blocks:      $TOTAL_BLOCKS"
echo "Assigned range:    [$START_BLOCK, $END_BLOCK)"
echo "Assigned blocks:   $BLOCK_COUNT"
echo "Frame files:       $FIRST_FRAME through $LAST_FRAME"
echo "Output directory:  $FRAMES_DIR"
echo "Host:              $(hostname)"
echo "Started:           $(date)"
echo

COMMAND=(
    python
    "$TIMESTAMP_SCRIPT"
    --sensor "$SENSOR_NAME"
    --render-dir "$RENDER_DIR"
    --output-dir "$OUTPUT_DIR"
    --render-fps "$RENDER_FPS"
    --start-block "$START_BLOCK"
    --end-block "$END_BLOCK"
    --hist-depth-min 0.02
    --hist-depth-max 4.0
    --hist-bins 32
    --no-progress
)

if [[ -n "$BLOCK_SIZE" ]]; then
    COMMAND+=(
        --block-size "$BLOCK_SIZE"
    )
fi

"${COMMAND[@]}"

echo
echo "Verifying task output..."

if [[ ! -s "$FIRST_FRAME_PATH" ]]; then
    echo "ERROR: First frame was not created:" >&2
    echo "$FIRST_FRAME_PATH" >&2
    exit 1
fi

if [[ ! -s "$LAST_FRAME_PATH" ]]; then
    echo "ERROR: Last frame was not created:" >&2
    echo "$LAST_FRAME_PATH" >&2
    exit 1
fi

echo "First output: $FIRST_FRAME_PATH"
echo "Last output:  $LAST_FRAME_PATH"
echo "Task completed successfully."
echo "Finished: $(date)"