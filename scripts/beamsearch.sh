#!/bin/bash
beam_size=4
beam_width=4
max_depth=10
mode='beamsearch'

while getopts w:s:d flag
do
    case "${flag}" in
        w) beam_width=${OPTARG};;
        s) beam_size=${OPTARG};;
        d) max_depth=${OPTARG};;
        m) mode=${OPTARG};;
    esac
done

NAME="${mode}_s${beam_size}_w${beam_width}_d${max_depth}";
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

python3 main_gsm8k.py search_algorithm=$mode search_algorithm.beam_size=$beam_size search_algorithm.beam_width=$beam_width \
                        search_algorithm.max_depth=$max_depth
EOT
