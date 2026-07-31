#!/bin/bash
#SBATCH --job-name=visual_summary
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=logs/visual_summary_%j.out
#SBATCH --error=logs/visual_summary_%j.err

set -euo pipefail

SCENE_NAME="${1:?Missing scene name}"
FRAME_INDEX="${2:-0}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"

INPUT_FILE="$PROJECT_DIR/outputs/$SCENE_NAME/timestamp_precomputed.npz"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME/figures/summary"
TTC_FILE="$OUTPUT_DIR/time_to_contact.npz"

cd "$PROJECT_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Visualization summary"
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Host:      $(hostname)"
echo "Scene:     $SCENE_NAME"
echo "Frame:     $FRAME_INDEX"
echo "Input:     $INPUT_FILE"
echo "Output:    $OUTPUT_DIR"
echo "Started:   $(date)"
echo

"$PYTHON" -m analysis.visualize_timestamps \
    --mode summary \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ttc-file "$TTC_FILE" \
    --frame "$FRAME_INDEX" \
    --ttc-window-ms 10 \
    --ttc-min-closing-speed 0.10 \
    --ttc-max-s 10 \
    --ttc-warning-s 1

echo
echo "Summary completed: $(date)"