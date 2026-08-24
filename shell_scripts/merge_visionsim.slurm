#!/bin/bash
#SBATCH --job-name=merge_visionsim
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem=2g
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%j_merge_visionsim.out
#SBATCH --error=logs/%j_merge_visionsim.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/merge_visionsim.slurm \
#       <scene_name> <first_frame> <last_frame>
#
# Example:
#   sbatch shell_scripts/merge_visionsim.slurm \
#       drone_flyby 1 240

SCENE_NAME="${1:?Missing scene name}"
FIRST_FRAME="${2:?Missing first frame}"
LAST_FRAME="${3:?Missing last frame}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
MERGE_SCRIPT="$PROJECT_DIR/shell_scripts/merge_visionsim_segments.sh"

cd "$PROJECT_DIR"

if [[ ! -x "$MERGE_SCRIPT" ]]; then
    echo "ERROR: Merge script is missing or not executable: $MERGE_SCRIPT" >&2
    exit 1
fi

OUTPUT_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"

echo "Merging VisionSIM segments"
echo "Scene: $SCENE_NAME"
echo "Frames: $FIRST_FRAME to $LAST_FRAME"
echo "Started: $(date)"

# Remove any previous merged output for this scene.
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Removing existing merged scene directory:"
    echo " $OUTPUT_DIR"
    rm -rf -- "$OUTPUT_DIR"
fi

"$MERGE_SCRIPT" \
    "$SCENE_NAME" \
    "$FIRST_FRAME" \
    "$LAST_FRAME"

echo "Merge completed successfully."
echo "Finished: $(date)"