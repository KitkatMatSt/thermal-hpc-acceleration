#!/bin/bash
#BSUB -J task10_fused_mask_profile
#BSUB -q c02613
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o output_fused_mask_profile_%J.txt
#BSUB -e error_fused_mask_profile_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
module load cuda/11.8

REPORT_NAME="task10_ultra_vFinal_${LSB_JOBID}"

nsys profile -t cuda --stats=true -o "$REPORT_NAME" python3 -u simu_cupy_fused_mask.py 20

nsys stats "$REPORT_NAME".nsys-rep > stats_fused_mask_summary.txt