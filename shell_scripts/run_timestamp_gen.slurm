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

# Usage:
#   sbatch shell_scripts/run_timestamp_gen.slurm <scene_name>
#
# Example:
#   sbatch shell_scripts/run_timestamp_gen.slurm drone_flyby
#
# Optional:
#   sbatch shell_scripts/run_timestamp_gen.slurm \
#       drone_flyby vl53l8ch 240

SCENE_NAME="${1:?Usage: sbatch $0 <scene_name> [sensor_name] [render_fps]}"
SENSOR_NAME="${2:-vl53l8ch}"
RENDER_FPS="${3:-240}"

PROJECT_DIR="$HOME/projects/timestamp_gen"
VENV="$PROJECT_DIR/.venv"

TIMESTAMP_SCRIPT="$PROJECT_DIR/timestamp_gen.py"
RENDER_DIR="$PROJECT_DIR/inputs/$SCENE_NAME"
OUTPUT_DIR="$PROJECT_DIR/outputs/$SCENE_NAME"

cd "$PROJECT_DIR"

source "$VENV/bin/activate"

export OPENCV_IO_ENABLE_OPENEXR=1

# Limit libraries that support CPU threading to the CPUs Slurm allocated.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "========================================"
echo "Timestamp generation job"
echo "========================================"
echo "Job ID:         ${SLURM_JOB_ID:-interactive}"
echo "Host:           $(hostname)"
echo "Started:        $(date)"
echo "Scene:          $SCENE_NAME"
echo "Sensor:         $SENSOR_NAME"
echo "Render FPS:     $RENDER_FPS"
echo "Render input:   $RENDER_DIR"
echo "Output:         $OUTPUT_DIR"
echo "Python:         $(command -v python)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-1}"
echo

python --version

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: Virtual-environment Python not found: $VENV/bin/python" >&2
    exit 1
fi

if [[ ! -f "$TIMESTAMP_SCRIPT" ]]; then
    echo "ERROR: Timestamp generator not found: $TIMESTAMP_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$RENDER_DIR/depths" ]]; then
    echo "ERROR: Depth directory not found: $RENDER_DIR/depths" >&2
    exit 1
fi

if [[ ! -d "$RENDER_DIR/normals" ]]; then
    echo "ERROR: Normal directory not found: $RENDER_DIR/normals" >&2
    exit 1
fi

DEPTH_COUNT=$(
    find "$RENDER_DIR/depths" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        | wc -l
)

NORMAL_COUNT=$(
    find "$RENDER_DIR/normals" \
        -maxdepth 1 \
        -type f \
        -name "*.exr" \
        | wc -l
)

echo "Depth files:  $DEPTH_COUNT"
echo "Normal files: $NORMAL_COUNT"

if (( DEPTH_COUNT < 2 )); then
    echo "ERROR: At least two depth EXR files are required." >&2
    exit 1
fi

if (( DEPTH_COUNT != NORMAL_COUNT )); then
    echo "ERROR: Depth and normal EXR counts do not match." >&2
    exit 1
fi

# Prevent stale files from an earlier run from being mistaken for new output.
if [[ -d "$OUTPUT_DIR" ]]; then
    BACKUP_DIR="${OUTPUT_DIR}_previous_${SLURM_JOB_ID:-manual}"
    echo "Moving existing output to: $BACKUP_DIR"
    mv "$OUTPUT_DIR" "$BACKUP_DIR"
fi

mkdir -p "$OUTPUT_DIR"

python "$TIMESTAMP_SCRIPT" \
    --sensor "$SENSOR_NAME" \
    --render-dir "$RENDER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --render-fps "$RENDER_FPS" \
    --no-full-dataset

PRECOMPUTED_FILE="$OUTPUT_DIR/timestamp_precomputed.npz"

if [[ ! -s "$PRECOMPUTED_FILE" ]]; then
    echo "ERROR: Expected output was not created: $PRECOMPUTED_FILE" >&2
    exit 1
fi

echo
echo "Timestamp generation completed successfully."
echo "Finished: $(date)"
echo

ls -lh "$OUTPUT_DIR"