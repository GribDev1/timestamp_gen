#!/bin/bash
#SBATCH --job-name=visionsim
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --array=0-7
#SBATCH --mail-type=NONE
#SBATCH --mail-user=grib6206@stthomas.edu
#SBATCH --output=logs/%A_visionsim_%a.out
#SBATCH --error=logs/%A_visionsim_%a.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/render_visionsim.slurm \
#       <scene_name>
#
# Example:
#   sbatch shell_scripts/render_visionsim.slurm \
#       drone_flyby
#
# The number of segments is controlled by:
#   #SBATCH --array=0-7
#
# This example creates eight array tasks.

SCENE_NAME="${1:?Missing scene name}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"
BLENDER_DIR="$HOME/software/blender-4.0.2-linux-x64"

SOURCE_BLEND="$PROJECT_DIR/blend_files/${SCENE_NAME}.blend"

SEGMENT_SCRIPT="$PROJECT_DIR/blender_scripts/create_segment_blend.py"
VALIDATION_SCRIPT="$PROJECT_DIR/shell_scripts/validate_visionsim_output.sh"

SEGMENT_ROOT="$PROJECT_DIR/inputs/${SCENE_NAME}_segments"
TEMP_ROOT="$PROJECT_DIR/tmp/visionsim_${SLURM_ARRAY_JOB_ID}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT}"

cd "$PROJECT_DIR"

source "$VENV/bin/activate"
export PATH="$VENV/bin:$BLENDER_DIR:$PATH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

unset DISPLAY
unset WAYLAND_DISPLAY
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

test -x "$VENV/bin/visionsim"
test -x "$BLENDER_DIR/blender"
test -s "$SOURCE_BLEND"
test -f "$SEGMENT_SCRIPT"
test -x "$VALIDATION_SCRIPT"

echo "Reading frame range from: $SOURCE_BLEND"

FRAME_RANGE=$(
    "$BLENDER_DIR/blender" \
        --background \
        "$SOURCE_BLEND" \
        --python-expr \
        'import bpy; print(f"DETECTED_FRAME_RANGE={bpy.context.scene.frame_start},{bpy.context.scene.frame_end}")' \
        2>/dev/null \
    | sed -n 's/^DETECTED_FRAME_RANGE=//p' \
    | tail -n 1
)

if [[ ! "$FRAME_RANGE" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "ERROR: Could not detect the Blender frame range." >&2
    echo "Detected value: $FRAME_RANGE" >&2
    exit 1
fi

IFS=',' read -r FIRST_FRAME LAST_FRAME <<< "$FRAME_RANGE"

if (( FIRST_FRAME < 1 )); then
    echo "ERROR: First frame must be at least 1." >&2
    exit 2
fi

if (( LAST_FRAME < FIRST_FRAME )); then
    echo "ERROR: Last frame must not be less than first frame." >&2
    exit 2
fi

echo "Detected Blender frame range: $FIRST_FRAME to $LAST_FRAME"

mkdir -p "$SEGMENT_ROOT"
mkdir -p "$TEMP_ROOT"

TOTAL_FRAMES=$((LAST_FRAME - FIRST_FRAME + 1))
FRAMES_PER_TASK=$(((TOTAL_FRAMES + TASK_COUNT - 1) / TASK_COUNT))
SEGMENT_START=$((FIRST_FRAME + TASK_ID * FRAMES_PER_TASK))
SEGMENT_END=$((SEGMENT_START + FRAMES_PER_TASK - 1))

if (( SEGMENT_END > LAST_FRAME )); then
    SEGMENT_END="$LAST_FRAME"
fi

if (( SEGMENT_START > LAST_FRAME )); then
    echo "Task $TASK_ID has no assigned frames."
    exit 0
fi

SEGMENT_NAME=$(
    printf \
        "segment_%03d_%06d_%06d" \
        "$TASK_ID" \
        "$SEGMENT_START" \
        "$SEGMENT_END"
)

TEMP_BLEND="$TEMP_ROOT/${SEGMENT_NAME}.blend"
SEGMENT_DIR="$SEGMENT_ROOT/$SEGMENT_NAME"

EXPECTED_COUNT=$((SEGMENT_END - SEGMENT_START + 1))

echo "========================================"
echo "VisionSIM segmented render"
echo "========================================"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:    $TASK_ID"
echo "Array task count: $TASK_COUNT"
echo "Host:             $(hostname)"
echo "Started:          $(date)"
echo "Scene:            $SCENE_NAME"
echo "Frames:           $SEGMENT_START to $SEGMENT_END"
echo "Expected frames:  $EXPECTED_COUNT"
echo "Source blend:     $SOURCE_BLEND"
echo "Temporary blend:  $TEMP_BLEND"
echo "Output directory: $SEGMENT_DIR"
echo "Python:           $(command -v python)"
echo "VisionSIM:        $(command -v visionsim)"
echo "Blender:          $(command -v blender)"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-1}"
echo

rm -rf "$SEGMENT_DIR"
mkdir -p "$SEGMENT_DIR"

echo "Creating temporary segmented blend file..."

blender \
    --background \
    "$SOURCE_BLEND" \
    --python "$SEGMENT_SCRIPT" \
    -- \
    "$TEMP_BLEND" \
    "$SEGMENT_START" \
    "$SEGMENT_END"

test -s "$TEMP_BLEND"

echo
echo "Starting VisionSIM render..."

visionsim blender.render-animation \
    "$TEMP_BLEND" \
    "$SEGMENT_DIR" \
    --render-config.depths \
    --render-config.normals \
    --render-config.no-debug \
    --render-config.width=320 \
    --render-config.height=320 \
    --render-config.device-type=cpu \
    --render-config.no-use-denoising \
    --render-config.max-samples=16 \
    --render-config.adaptive-threshold=0.05 \
    --render-config.no-allow-skips

echo
echo "VisionSIM command completed."
echo "Validating output..."

"$VALIDATION_SCRIPT" \
    "$SEGMENT_DIR" \
    "$EXPECTED_COUNT"

echo
echo "First generated files:"

find "$SEGMENT_DIR" \
    -maxdepth 2 \
    -type f \
    | sort \
    | head -n 20

echo
echo "Segment completed successfully."
echo "Finished: $(date)"