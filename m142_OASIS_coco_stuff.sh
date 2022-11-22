#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=usis_coco
#SBATCH --output=coco_b32_gpu_wavelet_0%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=titan_rtx:2
#SBATCH --qos=ebatch
#SBATCH --partition=empl
#SBATCH --nodes=1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
nvidia-smi
nvidia-smi topo --matrix
conda activate testenv
# Run your python code
CUDA_VISIBLE_DEVICES=0,1 python train.py --name oasis_unsupervised_coco_b16_pixelSPADE --dataset_mode coco --gpu_ids 0,1 \
--dataroot /data/public/cocostuff/dataset_sis --no_labelmix \
--batch_size 16 --model_supervision 0 --supervised_num 20 \
--Du_patch_size 64 --netDu wavelet  \
--netG 9 --channels_G 16 \
 --num_epochs 500 --continue_train

