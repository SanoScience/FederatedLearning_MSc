#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2


source ~/python3_6/bin/activate
# $1 contains server's node name; $2 holds a client's id.
echo $1 $2
python3 client_segmentation.py $1 $2