#!/bin/bash
#SBATCH --partition plgrid
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --time 12:00:00
#SBATCH --output log-%J.txt
#SBATCH -A plgsano2

unset PYTHONPATH
source ~/python3_6/bin/activate

python batch_download_zips.py

exit 0