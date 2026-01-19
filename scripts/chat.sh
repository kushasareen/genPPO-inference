#!/bin/bash

#SBATCH --partition=main
#SBATCH --job-name=chat
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_chat.txt"
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 simple_inference_chat.py search_algorithm=bestofn_grpo search_algorithm.top_k=32 search_algorithm.n=32 search_algorithm.policy_model="Qwen/Qwen2-Math-1.5B" search_algorithm.reward_model="Qwen/Qwen2-Math-1.5B" search_algorithm.output_path='/home/mila/k/kusha.sareen/scratch/genPPO/outputs/chat' search_algorithm.seed=0 search_algorithm.num_samples=-1 search_algorithm.max_tokens=1048 search_algorithm.dataset=math128