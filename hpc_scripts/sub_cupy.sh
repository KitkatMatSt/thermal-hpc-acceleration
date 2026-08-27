#!/bin/bash
#BSUB -J task9_cupy
#BSUB -q c02613
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"

#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o output_cupy_%J.txt
#BSUB -e error_cupy_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

module load cuda/11.8

python3 -u simu_cupy.py 20