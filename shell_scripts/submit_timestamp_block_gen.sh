#!/bin/bash

set -euo pipefail

# Usage:
#   ./shell_scripts/submit_timestamp_blocks_gen.sh \
#       <scene_name> [sensor_name] [render_fps] [blocks_per_task] [max_concurrent]
#
# Example:
#   ./shell_scripts/submit_timestamp_blocks_gen.sh \
#       drone_flyby vl53l8ch 240 1000 8

SCENE_NAME="${1:?Missing scene name}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"
BLOCKS_PER_TASK="${4:-1000}"
MAX_CONCURRENT="${5:-8}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
BLOCK_SLURM="$PROJECT_DIR/shell_scripts/run_timestamp_blocks.slurm"
DEPTH_DIR="$PROJECT_DIR/inputs/$SCENE_NAME/depths"
NORMAL_DIR="$PROJECT_DIR/inputs/$SCENE_NAME/normals"

cd "$PROJECT_DIR"

if [[ ! -f "$BLOCK_SLURM" ]]; then
    echo "ERROR: Slurm worker not found: $BLOCK_SLURM" >&2
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

mkdir -p logs

CALCULATION=$(
    python - \
        "$SENSOR_NAME" \
        "$DEPTH_DIR" \
        "$NORMAL_DIR" \
        "$RENDER_FPS" \
        "$BLOCKS_PER_TASK" <<'PY'
import math
import sys
from pathlib import Path

from configs.sensor_presets import get_sensor_preset

sensor_name = sys.argv[1]
depth_dir = Path(sys.argv[2])
normal_dir = Path(sys.argv[3])
render_fps = float(sys.argv[4])
blocks_per_task = int(sys.argv[5])

sensor = get_sensor_preset(sensor_name)

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
block_duration_s = sensor.block_size_L / sensor.laser_rate_hz

total_blocks = int(math.floor(scene_duration_s / block_duration_s))

if total_blocks <= 0:
    raise SystemExit("The scene produces no timestamp blocks.")

task_count = math.ceil(total_blocks / blocks_per_task)
last_task = task_count - 1

print(len(depth_files))
print(total_blocks)
print(task_count)
print(last_task)
PY
)

mapfile -t VALUES <<< "$CALCULATION"

RENDER_FRAME_COUNT="${VALUES[0]}"
TOTAL_BLOCKS="${VALUES[1]}"
TASK_COUNT="${VALUES[2]}"
LAST_TASK="${VALUES[3]}"

echo "=================================================="
echo "Timestamp block-array submission"
echo "=================================================="
echo "Scene:              $SCENE_NAME"
echo "Sensor:             $SENSOR_NAME"
echo "Render FPS:         $RENDER_FPS"
echo "Rendered frames:    $RENDER_FRAME_COUNT"
echo "Total blocks:       $TOTAL_BLOCKS"
echo "Blocks per task:    $BLOCKS_PER_TASK"
echo "Array tasks:        $TASK_COUNT"
echo "Array range:        0-$LAST_TASK"
echo "Maximum concurrent: $MAX_CONCURRENT"
echo