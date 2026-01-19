#!/bin/bash
expansion_temp=0.1
max_depth=20
temp=0.7
mode='rebase_math_adv'
top_k=32
verification_temp=1.0

while getopts e:k:d:t:v: flag
do
    case "${flag}" in
        e) expansion_temp=${OPTARG};;
        d) max_depth=${OPTARG};;
        k) top_k=${OPTARG};;
        t) temp=${OPTARG};;
        v) verification_temp=${OPTARG};;

    esac
done

NAME="${mode}_k${top_k}_e${expansion_temp}_d${max_depth}_t${temp}_v${verification_temp}";
echo $NAME
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=$NAME
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_$NAME.txt"
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --constraint=48gb

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 main_gsm8k.py search_algorithm=$mode search_algorithm.max_depth=$max_depth search_algorithm.expansion_temp=$expansion_temp search_algorithm.generation_temp=$temp search_algorithm.top_k=$top_k search_algorithm.verification_temp=$verification_temp
EOT
