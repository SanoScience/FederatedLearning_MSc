#!/bin/bash

#SBATCH --output=%j_server.txt
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -n 1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -A plgsano3-gpu

source $SCRATCH/anaconda3/etc/profile.d/conda.sh

conda activate ffcv
which python

echo $SLURM_JOB_NODELIST
python3 server_classification.py --c $1 --r $2 --m $3 --d $4 --le $5 --lr $6 --bs $7 --mf $8 --ff $9 --data-selection ${10} --hpc-log
