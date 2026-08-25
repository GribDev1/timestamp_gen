#!/bin/bash
#SBATCH --job-name=merge_visionsim
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --mem=2g
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%j_merge_visionsim.out
#SBATCH --error=logs/%j_merge_visionsim.err

set -euo pipefail

# Usage:
#   sbatch shell_scripts/merge_visionsim.slurm \
#       <scene_name>
#
# Example:
#   sbatch shell_scripts/merge_visionsim.slurm \
#       drone_flyby

SCENE_NAME="${1:?Missing scene name}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
MERGE_SCRIPT="$PROJECT_DIR/shell_scripts/merge_visionsim_segments.sh"
SEGMENT_ROOT="$PROJECT_DIR/inputs/${SCENE_NAME}_segments"
OUTPUT_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"

cd "$PROJECT_DIR"

if [[ ! -x "$MERGE_SCRIPT" ]]; then
    echo "ERROR: Merge script is missing or not executable: $MERGE_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$SEGMENT_ROOT" ]]; then
    echo "ERROR: Segment directory does not exist:" >&2
    echo "$SEGMENT_ROOT" >&2
    exit 1
fi

mapfile -t SEGMENT_NAMES < <(
    find "$SEGMENT_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'segment_*_*_*' \
        -printf '%f\n' \
        | sort
)

if (( ${#SEGMENT_NAMES[@]} == 0 )); then
    echo "ERROR: No VisionSIM segment directories found in:" >&2
    echo "$SEGMENT_ROOT" >&2
    exit 1
fi

FIRST_FRAME=$(
    printf '%s\n' "${SEGMENT_NAMES[@]}" |
        awk -F_ '
            NR == 1 { minimum = $3 + 0 }
            ($3 + 0) < minimum { minimum = $3 + 0 }
            END { print minimum }
        '
)

LAST_FRAME=$(
    printf '%s\n' "${SEGMENT_NAMES[@]}" |
        awk -F_ '
            NR == 1 { maximum = $4 + 0 }
            ($4 + 0) > maximum { maximum = $4 + 0 }
            END { print maximum }
        '
)

echo "Merging VisionSIM segments"
echo "Scene: $SCENE_NAME"
echo "Segments: ${#SEGMENT_NAMES[@]}"
echo "Frames: $FIRST_FRAME to $LAST_FRAME"
echo "Started: $(date)"

if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Removing existing merged scene directory:"
    echo "$OUTPUT_DIR"
    rm -rf -- "$OUTPUT_DIR"
fi

"$MERGE_SCRIPT" \
    "$SCENE_NAME" \
    "$FIRST_FRAME" \
    "$LAST_FRAME"

echo "Merge completed successfully."
echo "Finished: $(date)"