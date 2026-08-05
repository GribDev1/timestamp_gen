#!/bin/bash

set -euo pipefail

# Usage:
#   ./shell_scripts/submit_timestamp_blocks_gen.sh \
#       <scene_name> [sensor_name] [render_fps] \
#       [blocks_per_task] [max_concurrent] [block_size]
#
# Example:
#   ./shell_scripts/submit_timestamp_blocks_gen.sh \
#       drone_flyby vl53l8ch 240 1000 8 256

SCENE_NAME="${1:?Missing scene name}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"
BLOCKS_PER_TASK="${4:-1000}"
MAX_CONCURRENT="${5:-8}"
BLOCK_SIZE="${6:-}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

BLOCK_SLURM="$PROJECT_DIR/shell_scripts/run_timestamp_blocks.slurm"
MERGE_SLURM="$PROJECT_DIR/shell_scripts/merge_timestamp_block_shards.slurm"

DEPTH_DIR="$PROJECT_DIR/inputs/$SCENE_NAME/depths"
NORMAL_DIR="$PROJECT_DIR/inputs/$SCENE_NAME/normals"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$BLOCK_SLURM" ]]; then
    echo "ERROR: Slurm worker not found: $BLOCK_SLURM" >&2
    exit 1
fi

if [[ ! -f "$MERGE_SLURM" ]]; then
    echo "ERROR: Merge worker not found: $MERGE_SLURM" >&2
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

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: max_concurrent must be a positive integer." >&2
    exit 1
fi

if ! [[ "$RENDER_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: render_fps must be a positive number." >&2
    exit 1
fi

if [[ -n "$BLOCK_SIZE" ]] &&
   ! [[ "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: block_size must be a positive integer." >&2
    exit 1
fi

mkdir -p logs

CALCULATION=$(
    "$PYTHON" - \
        "$SENSOR_NAME" \
        "$DEPTH_DIR" \
        "$NORMAL_DIR" \
        "$RENDER_FPS" \
        "$BLOCKS_PER_TASK" \
        "$BLOCK_SIZE" <<'PY'
import math
import sys
from pathlib import Path

from configs.sensor_presets import get_sensor_preset

sensor_name = sys.argv[1]
depth_dir = Path(sys.argv[2])
normal_dir = Path(sys.argv[3])
render_fps = float(sys.argv[4])
blocks_per_task = int(sys.argv[5])
block_size_arg = sys.argv[6]

if render_fps <= 0:
    raise SystemExit("render_fps must be positive")

if blocks_per_task <= 0:
    raise SystemExit("blocks_per_task must be positive")

sensor = get_sensor_preset(sensor_name)

block_size = (
    int(block_size_arg)
    if block_size_arg
    else sensor.block_size_L
)

if block_size <= 0:
    raise SystemExit("block_size must be positive")

depth_files = sorted(depth_dir.glob("*.exr"))
normal_files = sorted(normal_dir.glob("*.exr"))

if len(depth_files) < 2:
    raise SystemExit(
        f"At least 2 depth EXRs are required; found {len(depth_files)}."
    )

if len(depth_files) != len(normal_files):
    raise SystemExit(
        "Depth/normal count mismatch: "
        f"{len(depth_files)} depth files and "
        f"{len(normal_files)} normal files."
    )

render_intervals = len(depth_files) - 1
scene_duration_s = render_intervals / render_fps

block_duration_s = (
    block_size / sensor.laser_rate_hz
)

total_blocks = int(
    math.floor(scene_duration_s / block_duration_s)
)

if total_blocks <= 0:
    raise SystemExit("The scene produces no timestamp blocks.")

task_count = math.ceil(
    total_blocks / blocks_per_task
)

last_task = task_count - 1

print(len(depth_files))
print(block_size)
print(total_blocks)
print(task_count)
print(last_task)
PY
)

mapfile -t VALUES <<< "$CALCULATION"

RENDER_FRAME_COUNT="${VALUES[0]}"
ACTIVE_BLOCK_SIZE="${VALUES[1]}"
TOTAL_BLOCKS="${VALUES[2]}"
TASK_COUNT="${VALUES[3]}"
LAST_TASK="${VALUES[4]}"

echo "=================================================="
echo "Timestamp block pipeline submission"
echo "=================================================="
echo "Scene:              $SCENE_NAME"
echo "Sensor:             $SENSOR_NAME"
echo "Render FPS:         $RENDER_FPS"
echo "Rendered frames:    $RENDER_FRAME_COUNT"
echo "Block size:         $ACTIVE_BLOCK_SIZE pulses"
echo "Total blocks:       $TOTAL_BLOCKS"
echo "Blocks per task:    $BLOCKS_PER_TASK"
echo "Array tasks:        $TASK_COUNT"
echo "Array range:        0-$LAST_TASK"
echo "Maximum concurrent: $MAX_CONCURRENT"
echo

BLOCK_SUBMISSION=$(
    sbatch \
        --parsable \
        --array="0-${LAST_TASK}%${MAX_CONCURRENT}" \
        "$BLOCK_SLURM" \
        "$SCENE_NAME" \
        "$SENSOR_NAME" \
        "$RENDER_FPS" \
        "$BLOCKS_PER_TASK" \
        "$ACTIVE_BLOCK_SIZE"
)

BLOCK_JOB_ID="${BLOCK_SUBMISSION%%;*}"

MERGE_SUBMISSION=$(
    sbatch \
        --parsable \
        --dependency="afterok:${BLOCK_JOB_ID}" \
        "$MERGE_SLURM" \
        "$SCENE_NAME" \
        "$TOTAL_BLOCKS"
)

MERGE_JOB_ID="${MERGE_SUBMISSION%%;*}"

echo "Submitted timestamp block array: $BLOCK_JOB_ID"
echo "Submitted precomputed merge job:  $MERGE_JOB_ID"
echo
echo "Raw frames will be written to:"
echo "  $PROJECT_DIR/outputs/$SCENE_NAME/frames"
echo
echo "Precomputed shards will be written to:"
echo "  $PROJECT_DIR/outputs/$SCENE_NAME/precomputed_shards"
echo
echo "Final merged dataset will be written to:"
echo "  $PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"