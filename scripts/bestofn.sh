#!/bin/bash
mode='bestofn_ckpt2'
max_depth=10
temp=0.7

while getopts n:t:d flag
do
    case "${flag}" in
        d) max_depth=${OPTARG};;
        t) temp=${OPTARG};;
    esac
done

NAME="${mode}_d${max_depth}_t${temp}";
echo $NAME
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=$NAME
#SBATCH --output="/home/mila/k/kusha.sareen/scratch/genPPO/logs/%j_$NAME.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module load anaconda
cd /home/mila/k/kusha.sareen/genPPO/genPPO-inference
conda activate genPPO

unset CUDA_VISIBLE_DEVICES

python3 main_gsm8k.py search_algorithm=$mode search_algorithm.max_depth=$max_depth search_algorithm.generation_temp=$temp
EOT
