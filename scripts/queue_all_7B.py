import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='qwen7B_genPPO_0.3')
parser.add_argument('--queue', action='store_true', default=False)
args = parser.parse_args()
model = args.model
num_jobs = 0

model_path = f'/home/mila/k/kusha.sareen/scratch/genPPO/{model}'
dataset = 'math128'
dataset_path = "/network/scratch/k/kusha.sareen/genPPO/data/math/test"
num_samples = -1

# Make output dir
output_dir = f'/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

K_max = 128
k = 2 # 2?

# run.sh should set both n and top_k to k and set both model paths to the right thing and have the correct args

# QUEUE Sampling
search_alg = 'bestofn_math'
command = f'bash scripts/run_7B.sh -m {model_path} -d {dataset} -p {dataset_path} -k {K_max} -s {search_alg} -o {output_dir} -n {num_samples} -e 0'
print(command)
if args.queue: os.system(command)
num_jobs += 1

# QUEUE Search
search_alg = 'rebase_math'
seeds = [0, 1, 2]

while k <= K_max:
    for seed in seeds:
        for adv in [True, False]:
            search_alg = 'rebase_math'
            if adv:
                search_alg = 'rebase_math_adv'
            if k == K_max:
                # command = f'bash scripts/run_main.sh -m {model_path} -d {dataset} -p {dataset_path} -k {k} -s {search_alg} -o {output_dir} -n {num_samples} -e {seed}'
                command = f'bash scripts/run_7B.sh -m {model_path} -d {dataset} -p {dataset_path} -k {k} -s {search_alg} -o {output_dir} -n {num_samples} -e {seed}'
            else:
                command = f'bash scripts/run_7B.sh -m {model_path} -d {dataset} -p {dataset_path} -k {k} -s {search_alg} -o {output_dir} -n {num_samples} -e {seed}'
            print(command)
            if args.queue: os.system(command)
            num_jobs += 1
    # QUEUE 
    k *= 2

print('NUM JOBS QUEUED:', num_jobs)
# TODO: Write a script that collects results for plotting