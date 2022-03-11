#!/bin/bash

#SBATCH --output=logs/surrogates.%A_%a.log
#SBATCH --partition=ci
#SBATCH --array=90-160
#SBATCH --cpus-per-task=1


pwd;hostname;date

#VALUES=({1000..2000})
#THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
#echo "Starting job $THISJOBVALUE"
#python3 -u create_trainingdata.py $THISJOBVALUE

python3 -u runner.py --index $SLURM_ARRAY_TASK_ID


date

