#!/bin/bash
#BSUB -J WallHeatingStatic
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"          
#BSUB -W 00:30  # set a longer time limit for Task 5
#BSUB -o logs/out_%J.log
#BSUB -e logs/err_%J.log

cd $LS_SUBCWD
mkdir -p logs

# initialize the Python environment path
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026 

# initialize the CSV file for results
RESULT_FILE="scaling_results.csv"
echo "cores,runtime" > $RESULT_FILE

# test with different numbers of cores
N=50 # reduce N to 50 for testing purposes

# run the benchmark for each core count
for p in 1 2 4 8 16
do
    echo "Running Task 5 with N=$N on p=$p cores..."
    # use /usr/bin/time -f "%e" to get just the wall time in seconds
    runtime=$( /usr/bin/time -f "%e" python simu_para_static.py $N $p 2>&1 > /dev/null )
    echo "$p,$runtime" >> $RESULT_FILE
done

echo "Benchmark complete. Results saved to $RESULT_FILE."