#!/bin/bash -l
# Slurm parameters
#SBATCH --job-name=Cycle
#SBATCH --output=cycle%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:30:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testenv
# Run your python code
python eval.py 