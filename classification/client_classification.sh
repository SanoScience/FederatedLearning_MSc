#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2


source ~/python3_6/bin/activate
# $1 contains server's node name; $2 holds a client's id.
echo $1 $2 $3
python3 client_classification.py $1 $2 $3
