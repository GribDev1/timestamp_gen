#!/bin/bash

set -euo pipefail

SCENE_NAME="${1:?Usage: $0 <scene_name> <first_frame> <last_frame>}"
FIRST_FRAME="${2:?Missing first frame}"
LAST_FRAME="${3:?Missing last frame}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
SEGMENT_ROOT="$PROJECT_DIR/inputs/${SCENE_NAME}_segments"
FINAL_ROOT="$PROJECT_DIR/inputs/${SCENE_NAME}"

FINAL_DEPTHS="$FINAL_ROOT/depths"
FINAL_NORMALS="$FINAL_ROOT/normals"

if [[ ! -d "$SEGMENT_ROOT" ]]; then
    echo "ERROR: Segment directory not found: $SEGMENT_ROOT" >&2
    exit 1
fi

# Avoid silently overwriting a previous completed render.
if [[ -e "$FINAL_ROOT" ]]; then
    echo "ERROR: Final output already exists: $FINAL_ROOT" >&2
    echo "Move or remove it before merging." >&2
    exit 1
fi

mkdir -p "$FINAL_DEPTHS" "$FINAL_NORMALS"

for segment_dir in "$SEGMENT_ROOT"/segment_*; do
    [[ -d "$segment_dir" ]] || continue

    echo "Merging: $segment_dir"

    for file in "$segment_dir"/depths/*.exr; do
        [[ -e "$file" ]] || continue

        destination="$FINAL_DEPTHS/$(basename "$file")"

        if [[ -e "$destination" ]]; then
            echo "ERROR: Duplicate depth filename: $destination" >&2
            exit 1
        fi

        cp "$file" "$destination"
    done

    for file in "$segment_dir"/normals/*.exr; do
        [[ -e "$file" ]] || continue

        destination="$FINAL_NORMALS/$(basename "$file")"

        if [[ -e "$destination" ]]; then
            echo "ERROR: Duplicate normal filename: $destination" >&2
            exit 1
        fi

        cp "$file" "$destination"
    done
done

EXPECTED_COUNT=$((LAST_FRAME - FIRST_FRAME + 1))

DEPTH_COUNT=$(
    find "$FINAL_DEPTHS" -maxdepth 1 -type f -name "*.exr" | wc -l
)

NORMAL_COUNT=$(
    find "$FINAL_NORMALS" -maxdepth 1 -type f -name "*.exr" | wc -l
)

echo "Expected frames: $EXPECTED_COUNT"
echo "Depth files:     $DEPTH_COUNT"
echo "Normal files:    $NORMAL_COUNT"

if (( DEPTH_COUNT != EXPECTED_COUNT )); then
    echo "ERROR: Incorrect number of merged depth files." >&2
    exit 1
fi

if (( NORMAL_COUNT != EXPECTED_COUNT )); then
    echo "ERROR: Incorrect number of merged normal files." >&2
    exit 1
fi

echo "Merge completed successfully:"
echo "$FINAL_ROOT"