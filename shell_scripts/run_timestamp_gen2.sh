#!/bin/bash

set -euo pipefail

# Usage:
#   ./shell_scripts/run_timestamp_gen2.sh \
#       <scene_name> [sensor_name] [render_fps] [tof_width] [tof_height]
#
# Example:
#   ./shell_scripts/run_timestamp_gen2.sh \
#       drone_flyby vl53l8ch 240 8 8

SCENE_NAME="${1:?Missing scene name}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"
TOF_WIDTH="${4:-8}"
TOF_HEIGHT="${5:-8}"

PROJECT_DIR="$HOME/projects/timestamp_gen"

PIXEL_SLURM="$PROJECT_DIR/shell_scripts/run_pixel_gen.slurm"
MERGE_SLURM="$PROJECT_DIR/shell_scripts/merge_timestamp_pixels.slurm"

cd "$PROJECT_DIR"
mkdir -p logs

PIXEL_COUNT=$((TOF_WIDTH * TOF_HEIGHT))
LAST_TASK=$((PIXEL_COUNT - 1))

# Limit concurrent tasks to avoid excessive shared-filesystem traffic.
MAX_CONCURRENT=8

PIXEL_JOB_ID=$(
    sbatch \
        --parsable \
        --array="0-${LAST_TASK}%${MAX_CONCURRENT}" \
        "$PIXEL_SLURM" \
        "$SCENE_NAME" \
        "$SENSOR_NAME" \
        "$RENDER_FPS" \
        "$TOF_WIDTH" \
        "$TOF_HEIGHT"
)

MERGE_JOB_ID=$(
    sbatch \
        --parsable \
        --dependency="afterok:${PIXEL_JOB_ID}" \
        "$MERGE_SLURM" \
        "$SCENE_NAME" \
        "$TOF_WIDTH" \
        "$TOF_HEIGHT"
)

echo "Timestamp pipeline submitted."
echo "Scene:             $SCENE_NAME"
echo "Pixel count:       $PIXEL_COUNT"
echo "Pixel array job:   $PIXEL_JOB_ID"
echo "Merge job:         $MERGE_JOB_ID"
echo "Max simultaneous:  $MAX_CONCURRENT"