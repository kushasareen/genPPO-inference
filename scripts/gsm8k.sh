#!/bin/bash
beam_size=4
beam_width=4
max_depth=10
input_path='/network/scratch/k/kusha.sareen/genPPO/data/gsm8k/test'
policy_model='ReasoningMila/genppo_init_ckpt'
device='cuda'
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

python3 main_gsm8k.py --input_path=$input_path --policy_model=$policy_model --reward_model=$reward_model \
                        --device=$device --beam_size=$beam_size --beam_width=$beam_width --max_depth=$max_depth \
                        --search_algorithm=$mode
EOT
