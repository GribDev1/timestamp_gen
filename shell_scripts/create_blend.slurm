#!/bin/bash
#SBATCH --job-name=create_blend
#SBATCH --time=00:00:30
#SBATCH --ntasks=1
#SBATCH --mem=4g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=grib6206@stthomas.edu
#SBATCH --output=logs/%j_create_blend.out
#SBATCH --error=logs/%j_create_blend.err

set -euxo pipefail

PROJECT_DIR="$HOME/projects/timestamp_gen"
BLENDER="$HOME/software/blender-4.0.2-linux-x64/blender"

# Usage:
# sbatch shell_scripts/create_blend.slurm \
#     <scene_name>
#
# Example:
# sbatch shell_scripts/create_blend.slurm \
#     drone_flyby

SCENE_NAME="${1:?Usage: sbatch create_blend.slurm <scene_name>}"

SCRIPT_PATH="$PROJECT_DIR/blender_scripts/create_${SCENE_NAME}_blend.py"
GENERATED_BLEND="$PROJECT_DIR/${SCENE_NAME}.blend"
FINAL_BLEND="$PROJECT_DIR/blend_files/${SCENE_NAME}.blend"

RENDER_DIR="$PROJECT_DIR/inputs/${SCENE_NAME}"
SEGMENT_DIR="$PROJECT_DIR/inputs/${SCENE_NAME}_segments"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"

cd "$PROJECT_DIR"

rm -rf -- "$RENDER_DIR" "$SEGMENT_DIR" "$OUTPUT_DIR"

mkdir -p \
    "$PROJECT_DIR/inputs" \
    "$PROJECT_DIR/outputs" \
    "$PROJECT_DIR/blend_files"

unset DISPLAY
unset WAYLAND_DISPLAY
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "Host: $(hostname)"
echo "Blender: $BLENDER"
echo "Generation script: $SCRIPT_PATH"
echo "Expected output: $GENERATED_BLEND"
echo "Final output: $FINAL_BLEND"

test -x "$BLENDER"
test -f "$SCRIPT_PATH"

"$BLENDER" --background --factory-startup --python "$SCRIPT_PATH"

test -s "$GENERATED_BLEND"
mv -f "$GENERATED_BLEND" "$FINAL_BLEND"

ls -lh "$FINAL_BLEND"