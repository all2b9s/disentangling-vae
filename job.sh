#!/usr/bin/bash

#SBATCH --job-name=barpar
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=7-00:00:00


python VAE_train.py 1 > d1.log 2>&1 &
python VAE_train.py 2 > d2.log 2>&1 &
python VAE_train.py 3 > d3.log 2>&1 &
python VAE_train.py 4 > d4.log 2>&1 &
python VAE_train.py 5 > d5.log 2>&1 &

wait
