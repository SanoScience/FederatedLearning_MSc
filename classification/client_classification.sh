#!/bin/bash

#SBATCH --output=%j_client.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2


source ~/python3_6/bin/activate
module load plgrid/apps/cuda/11.0

echo $1 $2 $3 $4
python3 client_classification.py --sa $1 --c_id $2 --c $3 --m $4