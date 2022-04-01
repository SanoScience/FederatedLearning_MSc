#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=02:20:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2
#SBATCH --ntasks-per-node=10

CURR_DIR=$PWD
PARENT_DIR="$(dirname "$CURR_DIR")"
echo $PARENT_DIR
export PYTHONPATH=$PARENT_DIR
source venv/bin/activate
# $1 -> server's node name; $2 -> client's id; $3 -> clients number
echo $1 $2 $3
python3 client_segmentation.py $1 $2 $3

