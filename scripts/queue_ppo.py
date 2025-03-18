import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--queue', action='store_true', default=False)
args = parser.parse_args()
num_jobs = 0
# scripts = ['run_unkill.sh', 'run_main.sh', 'run.sh', 'run.sh']
# scripts = ['run_unkill.sh', 'run_main.sh', 'run_main.sh', 'run.sh', 'run.sh']

# scripts = ['run_main.sh'] + ['run.sh'] * 10
scripts = ['run_main.sh', 'run_main.sh'] + ['run.sh'] * 10
# scripts = ['run.sh'] *  10

for model in ['qwen_ppo']:
    # for dataset in ['math128', 'aime']:
    for dataset in ['math128']:
    # for dataset in ['aime']:
        dataset_path = "None"
        model_path = f'/home/mila/k/kusha.sareen/scratch/genPPO/{model}'
        num_samples = -1
        max_tokens = 2048

        # Make output dir
        output_dir = f'/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}_{dataset}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        K_max = 512
        k = 2 # 2?

        # run.sh should set both n and top_k to k and set both model paths to the right thing and have the correct args

        # QUEUE Sampling
        search_alg = 'bestofn_ppo'
        script = scripts[num_jobs % len(scripts)]
        # if model == 'qwen_ppo':
        #     script = 'run_ppo.sh'
        #     search_alg = 'bestofn_qwen_ppo'

        command = f'bash scripts/{script} -m {model_path} -d {dataset} -p {dataset_path} -k {K_max} -s {search_alg} -o {output_dir} -n {num_samples} -t {max_tokens} -e 0 -i {model}'
        print(command)
        if args.queue: os.system(command)
        num_jobs += 1

print(f"Submitted {num_jobs} jobs")
