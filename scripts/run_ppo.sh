#!/bin/bash
seed=0;
max_tokens=512;

while getopts m:d:p:k:s:o:e:n:t: flag
do
    case "${flag}" in
        m) model_path=${OPTARG};;
        d) dataset=${OPTARG};;
        p) dataset_path=${OPTARG};;
        k) k=${OPTARG};;
        s) search_alg=${OPTARG};;
        o) output_dir=${OPTARG};;
        e) seed=${OPTARG};;
        n) num_samples=${OPTARG};;
        t) max_tokens=${OPTARG};;
    esac
done


NAME="${search_alg}_${dataset}_k${k}";
echo $NAME
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=$NAME
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_$NAME.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --constraint=48gb

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 simple_inference_ppo.py search_algorithm=$search_alg search_algorithm.top_k=$k search_algorithm.n=$k search_algorithm.policy_model=$model_path search_algorithm.reward_model=$model_path search_algorithm.input_path=$dataset_path search_algorithm.output_path=$output_dir search_algorithm.seed=$seed search_algorithm.num_samples=$num_samples search_algorithm.max_tokens=$max_tokens search_algorithm.dataset=$dataset
EOT
