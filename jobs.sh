#!/bin/bash

#SBATCH --output=logs/mmota.%A_%a.log
#SBATCH --partition=ci
#SBATCH --array=0-307
#SBATCH --cpus-per-task=1


pwd;hostname;date

. spack-env.sh


python3 -u runner.py --index $SLURM_ARRAY_TASK_ID


date

