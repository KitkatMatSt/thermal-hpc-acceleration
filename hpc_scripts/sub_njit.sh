#!/bin/bash
#BSUB -J PyHPC_task7    # Job Name 
#BSUB -q hpc                 
#BSUB -n 1                   
#BSUB -W 00:20               
#BSUB -R "rusage[mem=4GB]"    # Increased slightly for Numba overhead
#BSUB -R "select[model == XeonGold6226R]" # Ensuring repeatable results
#BSUB -o output_task7_%J.txt 
#BSUB -e error_task7_%J.err

# 1. Initialize Python environment
# It is better to use the course conda environment which includes Numba
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# 2. Run the Numba implementation
# We are still timing 20 floorplans for a direct comparison 
time python3 simu_numba.py 20