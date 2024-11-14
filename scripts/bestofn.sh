#!/bin/bash
n=8
mode='bestofn'
max_depth=10

while getopts w:d flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        d) max_depth=${OPTARG};;
    esac
done

NAME="${mode}_n${n}_d${max_depth}";
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

python3 main_gsm8k.py search_algorithm=$mode search_algorithm.n=$n search_algorithm.max_depth=$max_depth
EOT
