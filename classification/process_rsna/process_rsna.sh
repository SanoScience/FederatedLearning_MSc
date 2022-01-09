#!/bin/bash

#SBATCH --output=%j_server.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2

source ~/python3_6/bin/activate
module load plgrid/apps/cuda/11.0

echo $SLURM_JOB_NODELIST
python3 process_rsna.py
