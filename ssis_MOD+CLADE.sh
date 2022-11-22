#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=oasisM/C
#SBATCH --output=oasis%j.%N.out
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --qos=batch
#SBATCH --gpus=rtx_a5000:1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
export PATH="/usrhomes/s1434/anaconda3/envs/myenv/bin:/usrhomes/s1434/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
conda activate /usrhomes/s1434/anaconda3/envs/myenv
# Run your python code

python train_supervised.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --batch_size 1  \
--model_supervision 2 --netG 1 --channels_G 64 --num_epochs 500 \
--checkpoints_dir ./checkpoints_MOD+CLADE --apply_CLADE