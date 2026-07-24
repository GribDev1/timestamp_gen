#!/bin/bash
#SBATCH --job-name=visionsim_render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/visionsim_%j.out
#SBATCH --error=logs/visionsim_%j.err

set -euxo pipefail

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"
BLENDER_DIR="$HOME/software/blender-4.0.2-linux-x64"

BLEND_FILE="$PROJECT_DIR/blend_files/test_flat_moving.blend"
RENDER_DIR="$PROJECT_DIR/inputs/flat_moving"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export PATH="$VENV/bin:$BLENDER_DIR:$PATH"

unset DISPLAY
unset WAYLAND_DISPLAY

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "VisionSIM: $(which visionsim)"
echo "Blender: $(which blender)"

python --version
blender --version
visionsim --help >/dev/null

test -f "$BLEND_FILE"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/inputs"

if [ -e "$RENDER_DIR" ]; then
    echo "Render directory already exists:"
    echo "$RENDER_DIR"
    echo "Remove or rename it before submitting a new render."
    exit 1
fi

visionsim blender.render-animation \
    "$BLEND_FILE" \
    "$RENDER_DIR" \
    --render-config.depths \
    --render-config.normals \
    --render-config.no-debug \
    --render-config.width=160 \
    --render-config.height=160 \
    --render-config.device-type=cuda \
    --render-config.no-use-denoising \
    --render-config.max-samples=16 \
    --render-config.adaptive-threshold=0.05 \
    --render-config.no-allow-skips

echo "VisionSIM render completed."

find "$RENDER_DIR" -maxdepth 2 -type f | sort | head -50