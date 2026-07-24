<<<<<<< Updated upstream
#!/bin/bash
#SBATCH --job-name=create_blend
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/create_blend_%j.out
#SBATCH --error=logs/create_blend_%j.err

set -euxo pipefail

module purge
module load blender

PROJECT_DIR="$HOME/projects/timestamp_gen"
SCRIPT_PATH="$PROJECT_DIR/blender_scripts/create_moving_plane_blend.py"

cd "$PROJECT_DIR"
mkdir -p logs blend_files

unset DISPLAY
unset WAYLAND_DISPLAY

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "Host: $(hostname)"
echo "Blender: $(which blender)"
blender --version

blender \
    --background \
    --factory-startup \
    --python "$SCRIPT_PATH"

test -f "$PROJECT_DIR/test_flat_moving.blend"

mv -f \
    "$PROJECT_DIR/test_flat_moving.blend" \
    "$PROJECT_DIR/blend_files/test_flat_moving.blend"

ls -lh "$PROJECT_DIR/blend_files/test_flat_moving.blend"
=======
#!/bin/bash
#SBATCH --job-name=create_blend
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/create_blend_%j.out
#SBATCH --error=logs/create_blend_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=grib6206@stthomas.edu

set -euo pipefail

module purge
module load blender

PROJECT_DIR="$HOME/projects/timestamp_gen"
SCRIPT_PATH="$PROJECT_DIR/blender_scripts/create_moving_plane_blend.py"
BLEND_OUTPUT_DIR="$PROJECT_DIR/inputs"

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p "$BLEND_OUTPUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Blender: $(which blender)"
blender --version

blender --background --python "$SCRIPT_PATH"

mv -f \
    "$PROJECT_DIR/test_flat_moving.blend" \
    "$BLEND_OUTPUT_DIR/test_flat_moving.blend"

echo "Blend file created:"
ls -lh "$BLEND_OUTPUT_DIR/test_flat_moving.blend"

echo "Finished: $(date)"
>>>>>>> Stashed changes
