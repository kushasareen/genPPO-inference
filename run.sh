#!/bin/bash

input_path='/network/scratch/k/khang.ngo/gen_vineppo/data/gsm8k/test'
policy_model='ReasoningMila/genppo_init_ckpt'
device='cuda'

python3 main_gsm8k.py --input_path=$input_path --policy_model=$policy_model --reward_model=$reward_model \
                        --device=$device
