#!/bin/bash

#SBATCH --account=ehp
#SBATCH --time=00:20:00
#SBATCH --job-name=slurm_python_5
#SBATCH -o OUTPUT_FILES/slurm_python.o%j # Name of stdout output file
#SBATCH -e OUTPUT_FILES/slurm_python.e%j # Name of stderr error file
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --mail-type=ALL
#SBATCH --hint=compute_bound

ulimit -n 4096

# load modules
module purge
module load PrgEnv-gnu/6.0.10 cray-mpich/7.7.19 slurm/18.08.7-1 cray-python/3.9.7.1

# Activate python environment and run script
source /caldera/projects/usgs/hazards/ehp/istone/denali_gnu/python_test/calc_SA/bin/activate

python calcCGOF3.py

deactivate
