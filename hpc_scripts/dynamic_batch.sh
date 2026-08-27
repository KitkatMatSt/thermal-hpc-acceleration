#!/bin/bash
#BSUB -J WallHeatingDynamic
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"          
#BSUB -W 00:30  
#BSUB -o logs/out_dynamic_%J.log
#BSUB -e logs/err_dynamic_%J.log

cd $LS_SUBCWD
mkdir -p logs

# initialize the Python environment path
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026 

# initialize the CSV file for results
RESULT_FILE="scaling_results_dynamic.csv"
echo "cores,runtime" > $RESULT_FILE

# use the same N as Task 5 for a fair comparison
N=50 

# run the benchmark for each core count
for p in 1 2 4 8 16
do
    echo "Running Task 6 (Dynamic) with N=$N on p=$p cores..."
    
    # send $N $p to sys.argv[1] and sys.argv[2]
    runtime=$( /usr/bin/time -f "%e" python simu_para_dynamic.py $N $p 2>&1 > /dev/null )
    echo "$p,$runtime" >> $RESULT_FILE
done

echo "Dynamic Scheduling Benchmark complete. Results saved to $RESULT_FILE."