#!/bin/bash
#SBATCH --job-name=create_blend
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/create_blend_%j.out
#SBATCH --error=logs/create_blend_%j.err

set -euxo pipefail

PROJECT_DIR="$HOME/projects/timestamp_gen"
SCRIPT_PATH="$PROJECT_DIR/blender_scripts/create_moving_plane_blend.py"

# Replace this with your actual Blender executable path.
BLENDER="$HOME/software/blender-4.0.2-linux-x64/blender"

cd "$PROJECT_DIR"
mkdir -p logs blend_files

unset DISPLAY
unset WAYLAND_DISPLAY

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "Host: $(hostname)"
echo "Blender executable: $BLENDER"

test -x "$BLENDER"
test -f "$SCRIPT_PATH"

"$BLENDER" --version

"$BLENDER" \
    --background \
    --factory-startup \
    --python "$SCRIPT_PATH"

test -f "$PROJECT_DIR/test_flat_moving.blend"

mv -f \
    "$PROJECT_DIR/test_flat_moving.blend" \
    "$PROJECT_DIR/blend_files/test_flat_moving.blend"

ls -lh "$PROJECT_DIR/blend_files/test_flat_moving.blend"