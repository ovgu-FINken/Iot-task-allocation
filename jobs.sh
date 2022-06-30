#!/bin/bash

#SBATCH --output=logs/dmota.%A_%a.log
#SBATCH --partition=ci
#SBATCH --array=0-99
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#SBATCH --mail-user=dominik.weikert@ovgu.de
pwd;hostname;date

#VALUES=({1000..2000})
#THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
#echo "Starting job $THISJOBVALUE"
#python3 -u create_trainingdata.py $THISJOBVALUE

python3 -u runner.py --index $SLURM_ARRAY_TASK_ID


date

