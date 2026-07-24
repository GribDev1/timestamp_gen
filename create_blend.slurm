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
VENV="$PROJECT_DIR/.venv"
BLENDER="$HOME/software/blender-4.0.2-linux-x64/blender"

SCRIPT_PATH="$PROJECT_DIR/blender_scripts/create_moving_plane_blend.py"
GENERATED_BLEND="$PROJECT_DIR/test_flat_moving.blend"
FINAL_BLEND="$PROJECT_DIR/blend_files/test_flat_moving.blend"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

mkdir -p blend_files

unset DISPLAY
unset WAYLAND_DISPLAY
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Blender: $BLENDER"

test -x "$BLENDER"
test -f "$SCRIPT_PATH"

"$BLENDER" --version

"$BLENDER" \
    --background \
    --factory-startup \
    --python "$SCRIPT_PATH"

test -s "$GENERATED_BLEND"
mv -f "$GENERATED_BLEND" "$FINAL_BLEND"

ls -lh "$FINAL_BLEND"