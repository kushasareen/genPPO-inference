#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=dunno
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_dunno.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 main_gsm8k.py search_algorithm=bestofn_vineppo
