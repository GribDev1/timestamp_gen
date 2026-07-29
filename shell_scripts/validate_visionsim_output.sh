#!/bin/bash

set -euo pipefail

# Usage:
#   validate_visionsim_outputs.sh <render_dir> <expected_count>
#
# Example:
#   validate_visionsim_outputs.sh \
#       inputs/flat_moving_segments/segment_000_000001_000030 \
#       30

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <render_dir> <expected_count>" >&2
    exit 2
fi

RENDER_DIR="$1"
EXPECTED_COUNT="$2"

DEPTH_DIR="$RENDER_DIR/depths"
NORMAL_DIR="$RENDER_DIR/normals"

if ! [[ "$EXPECTED_COUNT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Expected count must be a nonnegative integer." >&2
    exit 2
fi

if [[ ! -d "$RENDER_DIR" ]]; then
    echo "ERROR: Render directory does not exist: $RENDER_DIR" >&2
    exit 1
fi

if [[ ! -d "$DEPTH_DIR" ]]; then
    echo "ERROR: Depth directory does not exist: $DEPTH_DIR" >&2
    exit 1
fi

if [[ ! -d "$NORMAL_DIR" ]]; then
    echo "ERROR: Normal directory does not exist: $NORMAL_DIR" >&2
    exit 1
fi

DEPTH_COUNT=$(
    find "$DEPTH_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        -print \
        | wc -l
)

NORMAL_COUNT=$(
    find "$NORMAL_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        -print \
        | wc -l
)

echo "VisionSIM output validation"
echo "Render directory: $RENDER_DIR"
echo "Expected frames:  $EXPECTED_COUNT"
echo "Depth EXRs:       $DEPTH_COUNT"
echo "Normal EXRs:      $NORMAL_COUNT"

if (( DEPTH_COUNT != EXPECTED_COUNT )); then
    echo \
        "ERROR: Expected $EXPECTED_COUNT depth EXRs, found $DEPTH_COUNT." \
        >&2
    exit 1
fi

if (( NORMAL_COUNT != EXPECTED_COUNT )); then
    echo \
        "ERROR: Expected $EXPECTED_COUNT normal EXRs, found $NORMAL_COUNT." \
        >&2
    exit 1
fi

if (( DEPTH_COUNT != NORMAL_COUNT )); then
    echo "ERROR: Depth and normal EXR counts do not match." >&2
    exit 1
fi

EMPTY_DEPTH_COUNT=$(
    find "$DEPTH_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        -size 0 \
        -print \
        | wc -l
)

EMPTY_NORMAL_COUNT=$(
    find "$NORMAL_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        -size 0 \
        -print \
        | wc -l
)

if (( EMPTY_DEPTH_COUNT > 0 )); then
    echo "ERROR: Found $EMPTY_DEPTH_COUNT empty depth EXR files." >&2
    exit 1
fi

if (( EMPTY_NORMAL_COUNT > 0 )); then
    echo "ERROR: Found $EMPTY_NORMAL_COUNT empty normal EXR files." >&2
    exit 1
fi

echo "Validation passed."