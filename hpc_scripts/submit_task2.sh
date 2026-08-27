#!/bin/bash
#BSUB -J PyHPC_Task2         # Job Name
#BSUB -q hpc                 # Queue (hpc is usually the default)
#BSUB -n 1                   # Number of cores (start with 1 for reference)
#BSUB -W 00:30               # Walltime (HH:MM) - 30 mins is plenty for 20 floors
#BSUB -R "rusage[mem=2GB]"   # Memory request
#BSUB -o output_%J.txt       # Output file (%J will be replaced by the Job ID)
#BSUB -e error_%J.txt        # Error file

# 1. Load the necessary modules
# Load the NumPy module that matches Python 3.10.12
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
module load numpy/1.24.3-python-3.10.12-openblas-0.3.23

# 2. Run the reference implementation for 20 floorplans
# We use 'time' to get the total execution time
time python3 simulate.py 20