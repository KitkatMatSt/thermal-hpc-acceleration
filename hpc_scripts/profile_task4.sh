#!/bin/bash
#BSUB -J PyHPC_kernprof             # Job Name
#BSUB -q hpc                        # Queue (hpc is usually the default)
#BSUB -n 1                          # Number of cores (start with 1 for reference)
#BSUB -W 00:30                      # Walltime (HH:MM) - 30 mins is plenty for 1 floor
#BSUB -R "select[model == XeonGold6226R]" # Ensuring repeatable results
#BSUB -R "rusage[mem=2GB]"          # Memory request
#BSUB -o output_%J.txt              # Output file (%J will be replaced by the Job ID)
#BSUB -e error_%J.txt               # Error file

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Load the NumPy module that matches Python 3.10.12
module load numpy/1.24.3-python-3.10.12-openblas-0.3.23

# load line_profiler for memory profiling
pip install --user line_profiler

# Using the direct path to the kernprof executable we found earlier
/zhome/40/8/123523/.local/bin/kernprof -l -v simulate_kernprof.py 1
