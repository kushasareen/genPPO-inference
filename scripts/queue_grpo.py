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

# for model in ['qwen_ORM_4', 'qwen_ORM_8', 'qwen_ORM_16', 'qwen_ORM_1', 'qwen_ORM_2']:
# for model in ['clf_0.2', 'clf_1']:
# for model in ['grpo_math_0.01_42', 'grpo_math_0_42', 'grpo_math_0.5_42', 'grpo_math_0.1_42']:
# for model in ['qwen_genPPO_10', 'qwen_ppo']:
# for model in ['grpo_math_0.1_sft_42']:
# for model in ['grpo_scot_math_0_sft_42', 'grpo_scot_math_0,1_sft_42', 'grpo_scot_math_0.1_clf_42', 'grpo_scot_math_0.05_sft_42']:
# for model in ['grpo_scot_math_0.1_sft_1741909026_42', 'grpo_scot_math_0.1_sft_1741916318_42', 'grpo_scot_math_0.05_sft_5e-5_1741927164_42']:
# for model in ['grpo_scot_math_0.1_sft_5e-5_1741975298_42']:
# for model in ['grpo_scot_math_1_sft_5e-5_1742029900_42', 'grpo_scot_math_0.8_sft_5e-5_1742040760_42', 'grpo_scot_math_0.3_sft_5e-5_1742019040_42', 'grpo_scot_math_0.5_sft_3e-5_1741959288_42', 'grpo_scot_math_0.2_sft_5e-5_1742008179_42', 'grpo_scot_math_0.1_sft_5e-5_1741975298_42']:
# for model in ['grpo_scot_math_1_clf_3e-4__42']:
# for model in ['grpo_scot_math_1_sft_2e-4__42']:
# for model in ['grpo_scot_math_1_clf_1.5e-4__42']:
for model in ['grpo_scot_math_1_clf_3e-4__42']:
    # for dataset in ['math128', 'aime']:
    for dataset in ['math128']:
    # for dataset in ['aime']:
        dataset_path = "None"
        # model_path = f'/home/mila/k/kusha.sareen/scratch/genPPO/{model}'
        model_path = f'/home/mila/k/kusha.sareen/scratch/genPPO/nano_reasoning/nano_outputs/{model}/model'
        # model_path = f'/home/mila/k/kusha.sareen/scratch/genPPO/nano_reasoning/nano_outputs/{model}/best_model'
        num_samples = -1
        max_tokens = 2048

        # Make output dir
        output_dir = f'/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}_{dataset}'
        # output_dir = f'/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}_{dataset}_best'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        K_max = 512
        k = 2 # 2?

        # run.sh should set both n and top_k to k and set both model paths to the right thing and have the correct args

        # QUEUE Sampling
        search_alg = 'bestofn_grpo'
        script = scripts[num_jobs % len(scripts)]
        # if model == 'qwen_ppo':
        #     script = 'run_ppo.sh'
        #     search_alg = 'bestofn_qwen_ppo'

        command = f'bash scripts/{script} -m {model_path} -d {dataset} -p {dataset_path} -k {K_max} -s {search_alg} -o {output_dir} -n {num_samples} -t {max_tokens} -e 0 -i {model}'
        print(command)
        if args.queue: os.system(command)
        num_jobs += 1

print(f"Submitted {num_jobs} jobs")
