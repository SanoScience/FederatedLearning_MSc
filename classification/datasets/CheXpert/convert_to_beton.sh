#!/bin/bash

#SBATCH --output=%j_convert_job.txt
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -n 1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ffcv
module load plgrid/apps/cuda/11.0

python3 convert_to_beton.py
