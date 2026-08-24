#!/bin/bash
#SBATCH --job-name=visual_global_gif
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --output=logs/%A_visual_global_gif_%a.out
#SBATCH --error=logs/%A_visual_global_gif_%a.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
GIF_STRIDE="${2:-100}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script requires an array task ID}"

case "$TASK_ID" in
    0)
        MODE="depth-gif"
        ;;
    1)
        MODE="valid-gif"
        ;;
    *)
        echo "ERROR: Expected array task 0 or 1, got $TASK_ID" >&2
        exit 1
        ;;
esac

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

INPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/gifs"

cd "$PROJECT_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Global GIF visualization"
echo "========================================"
echo "Mode:       $MODE"
echo "Scene:      $SCENE_NAME"
echo "GIF stride: $GIF_STRIDE"
echo "Output:     $OUTPUT_DIR"
echo "Started:    $(date)"
echo

"$PYTHON" -m analysis.visualize_timestamps \
    --mode "$MODE" \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --gif-fps 12 \
    --gif-stride "$GIF_STRIDE"

echo
echo "$MODE completed: $(date)"