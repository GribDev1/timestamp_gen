#!/bin/bash
#SBATCH --job-name=normal_gif
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/normal_gif_%j.out
#SBATCH --error=logs/normal_gif_%j.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
FPS="${2:-24}"
FRAME_STRIDE="${3:-1}"
SCALE="${4:-2}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

NORMAL_DIR="$PROJECT_DIR/inputs/$SCENE_NAME/normals"
OUTPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/figures/normals/normals.gif"

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p "$(dirname "$OUTPUT_FILE")"

export OPENCV_IO_ENABLE_OPENEXR=1
export MPLBACKEND=Agg

"$PYTHON" analysis/create_normals_gif.py \
    --normal-dir "$NORMAL_DIR" \
    --output "$OUTPUT_FILE" \
    --fps "$FPS" \
    --stride "$FRAME_STRIDE" \
    --scale "$SCALE"

echo "Saved: $OUTPUT_FILE"