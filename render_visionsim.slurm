#!/bin/bash
#SBATCH --job-name=visionsim_render
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/visionsim_%j.out
#SBATCH --error=logs/visionsim_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"
BLENDER_DIR="$HOME/software/blender-4.0.2-linux-x64"

SCENE_NAME="flat_moving"

BLEND_FILE="$PROJECT_DIR/blend_files/test_flat_moving.blend"
RENDER_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"

cd "$PROJECT_DIR"

source "$VENV/bin/activate"

export PATH="$VENV/bin:$BLENDER_DIR:$PATH"

unset DISPLAY
unset WAYLAND_DISPLAY

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(command -v python)"
echo "VisionSIM: $(command -v visionsim)"
echo "Blender: $(command -v blender)"

python --version
blender --version

test -x "$(command -v blender)"
test -x "$(command -v visionsim)"
test -s "$BLEND_FILE"

mkdir -p "$PROJECT_DIR/inputs"

rm -rf "$RENDER_DIR"
mkdir -p "$RENDER_DIR"

visionsim blender.render-animation \
    "$BLEND_FILE" \
    "$RENDER_DIR" \
    --render-config.depths \
    --render-config.normals \
    --render-config.no-debug \
    --render-config.width=160 \
    --render-config.height=160 \
    --render-config.device-type=cpu \
    --render-config.no-use-denoising \
    --render-config.max-samples=16 \
    --render-config.adaptive-threshold=0.05 \
    --render-config.no-allow-skips

echo "VisionSIM command completed."

DEPTH_DIR="$RENDER_DIR/depths"
NORMAL_DIR="$RENDER_DIR/normals"

test -d "$DEPTH_DIR"
test -d "$NORMAL_DIR"

DEPTH_COUNT=$(
    find "$DEPTH_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        | wc -l
)

NORMAL_COUNT=$(
    find "$NORMAL_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        | wc -l
)

echo "Depth EXRs: $DEPTH_COUNT"
echo "Normal EXRs: $NORMAL_COUNT"

if [ "$DEPTH_COUNT" -lt 2 ]; then
    echo "ERROR: VisionSIM produced fewer than two depth EXRs."
    exit 1
fi

if [ "$DEPTH_COUNT" -ne "$NORMAL_COUNT" ]; then
    echo "ERROR: Depth and normal EXR counts do not match."
    exit 1
fi

echo "First generated files:"

find "$RENDER_DIR" \
    -maxdepth 2 \
    -type f \
    | sort \
    | sed -n '1,50p'

echo "VisionSIM render completed successfully."
echo "Finished: $(date)"