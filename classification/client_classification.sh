#!/bin/bash

#SBATCH --output=%j_client.txt
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -n 1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH -A plgsano3-gpu

source $SCRATCH/anaconda3/etc/profile.d/conda.sh
conda activate ffcv

echo $1 $2 $3 $4
python3 client_classification.py --sa $1 --c_id $2 --c $3 --d $4
