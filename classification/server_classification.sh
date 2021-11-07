#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2

source ~/python3_6/bin/activate
echo $SLURM_JOB_NODELIST
python3 server_segmentation.py
