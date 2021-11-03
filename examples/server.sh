#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH -A plgsano2
#SBATCH --gres=gpu:1

source venv/bin/activate
echo $SLURM_JOB_NODELIST
python3 server_pt.py 
