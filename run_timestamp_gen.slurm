#!/bin/bash
#SBATCH --job-name=timestamp_gen
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --output=logs/timestamp_%j.out
#SBATCH --error=logs/timestamp_%j.err

set -euo pipefail

module purge
module load python3/3.11.9

cd "$HOME/projects/timestamp_gen"
source .venv/bin/activate

mkdir -p logs

python timestamp_gen.py \
    --sensor vl53l8ch \
    --render-dir inputs/flat_moving \
    --output-dir outputs/flat_moving \
    --render-fps 240 \
