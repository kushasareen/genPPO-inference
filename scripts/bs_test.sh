#!/bin/bash

#SBATCH --partition=main
#SBATCH --job-name=bs_test
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_bs_test.txt"
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 main_gsm8k.py search_algorithm=beamsearch_math search_algorithm.seed=0 search_algorithm.num_samples=250
