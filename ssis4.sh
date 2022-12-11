#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=oa1_2t
#SBATCH --output=oasis1_2scale_tanh__b2_no_noise_injection_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
# SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --qos=batch
#SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=rtx_2080ti


# Activate everything you need
#echo $PYENV_ROOT
#echo $PATHcannot assign 'torch.cuda.FloatTensor' as parameter 'weight' (torch.nn.Parameter or None expected)
export PATH="/usrhomes/s1434/anaconda3/envs/myenv/bin:/usrhomes/s1434/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
conda activate /usrhomes/s1434/anaconda3/envs/myenv
# Run your python code

python train_supervised.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --batch_size 2  \
--model_supervision 2 --netG 4 --channels_G 64 --num_epochs 500 \
--checkpoints_dir ./checkpoints1 --Matrix_Computation