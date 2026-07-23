#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --ntasks=2
#SBATCH --mem=80g
#SBATCH --mail-type=None
#SBATCH --mail-user=grib6206@stthomas.edu

# export LDFLAGS="-L/usr/lib64 -lblas -lcblas"
# export CFLAGS="-I/usr/include/cblas"
# export BLAS_H=/usr/include/cblas
# module load mkl
# module load gcc

module load python3/3.11.9