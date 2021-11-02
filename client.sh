#!/bin/bash



#SBATCH --output=%j.txt

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH -A plgsano2


source venv/bin/activate

echo $1
python3 client_pt.py $1
