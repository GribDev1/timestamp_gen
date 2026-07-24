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

PROJECT_DIR="$HOME/projects/timestamp_gen"

cd "$PROJECT_DIR"
mkdir -p logs

source .venv/bin/activate

echo "Python: $(which python)"
python --version

python timestamp_gen.py \
    --sensor vl53l8ch \
    --render-dir inputs/flat_moving \
    --output-dir outputs/flat_moving \
    --render-fps 240 \
    --no-full-dataset