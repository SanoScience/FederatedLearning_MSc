#!/bin/bash

#SBATCH --output=%j_server.txt
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2

source ~/anaconda3/etc/profile.d/conda.sh

conda activate ffcv
which python
module load plgrid/apps/cuda/11.0

echo $SLURM_JOB_NODELIST
python3 server_classification.py --c $1 --r $2 --m $3 --d $4 --le $5 --lr $6 --bs $7 --mf $8 --ff $9 --hpc-log --downsample-test
