#!/bin/bash

#SBATCH --output=%j.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH -A plgsano2

#SBATCH --gres=gpu:1

# args: 1. clients 2. rounds 3. aggregation strategy 4. local epochs
# sample call: ./server.sh 5 10 FedAdam 1

source venv/bin/activate
CURR_DIR=$PWD
PARENT_DIR="$(dirname "$CURR_DIR")"
echo $PARENT_DIR
export PYTHONPATH=$PARENT_DIR

echo $SLURM_JOB_NODELIST
python3 server_segmentation.py --c $1 --r $2 --a $3 --le $4 --lr $5
