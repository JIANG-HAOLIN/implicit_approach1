#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=gtavtocc
#SBATCH --output=gtavtocityscapes_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=titan_rtx:4
#SBATCH --qos=ebatch
#SBATCH --partition=empl
#SBATCH --nodes=linse20

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
nvidia-smi
conda activate testenv
# Run your python code
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name oasis_unsupervised_gtavtocityscapes_B32_ch64 --dataset_mode gtavtocityscapes --gpu_ids 0,1,2,3 \
--dataroot /data/public/cityscapes --no_labelmix \
--batch_size 16 --model_supervision 0 --supervised_num 20 \
--Du_patch_size 64 --netDu wavelet  \
--netG 0 --channels_G 64 \
 --num_epochs 500



