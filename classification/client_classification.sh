#!/bin/bash

#SBATCH --output=%j_client.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2


source ~/python3_6/bin/activate
module load plgrid/apps/cuda/11.0

# $1 contains server's node name; $2 holds a client's id.
echo $1 $2 $3
python3 client_classification.py $1 $2 $3
