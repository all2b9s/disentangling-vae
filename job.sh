#!/usr/bin/bash

#SBATCH --job-name=barpar
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=7-00:00:00


mamba activate pytorch

num_z=2
sampling=mws
echo num_z = $numz
echo sampling = $sampling

hostname; pwd; date

time python VAE_train.py $num_z $sampling

date
