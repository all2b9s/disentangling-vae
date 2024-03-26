#!/usr/bin/bash

#SBATCH --job-name=barpar
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=8
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00


source /mnt/home/yinli/mamba/bin/activate pytorch

[[ -z $num_z ]] && exit 1
echo num_z = $num_z

hostname; pwd; date

time srun python VAE_train.py $num_z

date
