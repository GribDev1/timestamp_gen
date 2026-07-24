#!/bin/bash
#SBATCH --job-name=timestamp_gen
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --output=logs/timestamp_%j.out
#SBATCH --error=logs/timestamp_%j.err

set -euxo pipefail

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

RENDER_DIR="$PROJECT_DIR/inputs/flat_moving"
OUTPUT_DIR="$PROJECT_DIR/outputs/flat_moving"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

export OPENCV_IO_ENABLE_OPENEXR=1

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version

test -d "$RENDER_DIR/depths"
test -d "$RENDER_DIR/normals"

DEPTH_COUNT=$(find "$RENDER_DIR/depths" -maxdepth 1 -name "*.exr" | wc -l)
NORMAL_COUNT=$(find "$RENDER_DIR/normals" -maxdepth 1 -name "*.exr" | wc -l)

echo "Depth files: $DEPTH_COUNT"
echo "Normal files: $NORMAL_COUNT"

if [ "$DEPTH_COUNT" -lt 2 ]; then
    echo "At least two depth files are required."
    exit 1
fi

if [ "$DEPTH_COUNT" -ne "$NORMAL_COUNT" ]; then
    echo "Depth and normal file counts do not match."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python timestamp_gen.py \
    --sensor vl53l8ch \
    --render-dir "$RENDER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --render-fps 240 \
    --no-full-dataset

test -s "$OUTPUT_DIR/timestamp_precomputed.npz"

echo "Timestamp generation completed."
ls -lh "$OUTPUT_DIR"