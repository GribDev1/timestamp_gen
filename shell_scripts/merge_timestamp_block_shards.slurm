#!/bin/bash
#SBATCH --job-name=timestamp_merge
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/timestamp_merge_%j.out
#SBATCH --error=logs/timestamp_merge_%j.err

set -euo pipefail

# Usage:
#   sbatch merge_timestamp_block_shards.slurm \
#       <scene_name> [expected_blocks]

SCENE_NAME="${1:?Missing scene name}"
EXPECTED_BLOCKS="${2:-}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
PYTHON="$PROJECT_DIR/.venv/bin/python"
MERGE_SCRIPT="$PROJECT_DIR/scripts/merge_timestamp_block_shards.py"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"

cd "$PROJECT_DIR"

COMMAND=(
    "$PYTHON"
    "$MERGE_SCRIPT"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$EXPECTED_BLOCKS" ]]; then
    COMMAND+=(
        --expected-blocks "$EXPECTED_BLOCKS"
    )
fi

echo "Scene:           $SCENE_NAME"
echo "Output:          $OUTPUT_DIR"
echo "Expected blocks: ${EXPECTED_BLOCKS:-not supplied}"
echo "Host:            $(hostname)"
echo "Started:         $(date)"

"${COMMAND[@]}"

if [[ ! -s "$OUTPUT_DIR/timestamp_precomputed.npz" ]]; then
    echo "ERROR: timestamp_precomputed.npz missing." >&2
    exit 1
fi

echo "Merge completed."
echo "Finished: $(date)"