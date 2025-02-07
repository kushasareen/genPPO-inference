import argparse
import gc
from reward_model import GenVinePPOVerifier
from tree import TreeNode
from verify_gsm8k import evaluate_predictions, estimate_token_count_at_k, estimate_time_at_k
import time
from utils import get_search_tree_and_generator, load_inference_dataset, load_model, save_results, get_reward_model, load_ppo_model, get_all_models, get_ppo_avg_orm_score, get_question, log_everything, parse_top_nodes
import asyncio
import hydra
import numpy as np
from omegaconf import OmegaConf
from eval import run_evals

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_inference_dataset(args)
    llm, sampling_params, reward_model = get_all_models(args)
    asyncio.run(run_inference(llm, reward_model, sampling_params, dataset, args))


async def run_inference(llm, reward_model, sampling_params, dataset, args):
    start = time.time()
    
    if args.llm_as_judge:
        tasks, node_generator = collect_tasks_orm(args, dataset, llm, reward_model, sampling_params)
    else:
        tasks, node_generator = collect_tasks_search(args, dataset, llm, reward_model, sampling_params)

    all_top_nodes = [await task for task in tasks]

    time_taken = time.time() - start
    total_tokens = node_generator.token_count + reward_model.token_count
    all_preds, all_different_scores = await parse_top_nodes(args, all_top_nodes, reward_model)

    log_everything(all_preds, all_different_scores, time_taken, all_top_nodes, total_tokens, args)
    run_evals(all_preds, all_different_scores, time_taken, total_tokens, args)


def collect_tasks_search(args, dataset, llm, reward_model, sampling_params):
    tasks = []
    if args.num_samples == -1:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples

    for i in range(num_samples):
        question = get_question(dataset, i, args)
        prompt = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n' # prompt should match training data format
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {"sum": 0, "min": 0, "last": 0} if args.log_all_scores else {args.aggregator: 0}) # 0 = 1 for logprobs
        tree, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        tasks.append(asyncio.create_task(tree.search(generate_children=node_generator, max_depth=args.max_depth)))
        gc.collect()

    return tasks, node_generator

def collect_tasks_orm(args, dataset, llm, reward_model, sampling_params):
    tasks = []

    if args.num_samples == -1:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples

    for i in range(num_samples):
        question = get_question(dataset, i, args)
        prompt = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n' # prompt should match training data format
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {args.aggregator: 0}) # 0 = 1 for logprobs
        
        _, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        tasks.append(asyncio.create_task(node_generator(root, width=args.top_k)))
        gc.collect()

    return tasks, node_generator


if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
