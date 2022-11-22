#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=usis_ade20k
#SBATCH --output=ade20k_unsupervised%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=128G
#SBATCH --gpus=titan_rtx:4
#SBATCH --qos=ebatch
#SBATCH --partition=empl
#SBATCH --nodes=1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
nvidia-smi
conda activate testenv
# Run your python code
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name oasis_unsupervised_ade20k_b32_pixelSPADE --dataset_mode ade20k --gpu_ids 0,1,2,3 \
--dataroot /data/public/ade20k/data/ADEChallengeData2016 --no_labelmix \
--batch_size 32 --model_supervision 0 --supervised_num 20 \
--Du_patch_size 64 --netDu wavelet  \
--netG 9 --channels_G 16 \
 --num_epochs 500

