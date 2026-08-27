#!/bin/bash
#BSUB -J PyHPC_task8_V100
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"   
#BSUB -gpu "num=1"            
#BSUB -W 00:30
#BSUB -o output_V100_%J.txt
#BSUB -e error_V100_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026


module load cuda/11.8


python3 -u simu_gpu.py 20