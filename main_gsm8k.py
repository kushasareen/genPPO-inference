import argparse
import gc
from reward_model import GenVinePPOVerifier
from tree import TreeNode
from verify_gsm8k import evaluate_predictions, estimate_token_count_at_k, estimate_time_at_k
import time
from utils import get_search_tree_and_generator, load_inference_dataset, load_model, save_results, get_reward_model, load_ppo_model, get_all_models, get_ppo_avg_orm_score, get_question, log_everything, parse_top_nodes, log_solution, get_num_samples, get_prompt, get_task, log_args
import asyncio
import hydra
import numpy as np
from omegaconf import OmegaConf
from eval import run_evals
import os

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_inference_dataset(args)
    llm, sampling_params, reward_model, tokenizer = get_all_models(args)
    if args.use_async:
        asyncio.run(run_inference(llm, reward_model, sampling_params, dataset, tokenizer, args))
    else:
        # open folder corresponding to the current run and read status.txt to get the start_sample
        # need to open tokens_so_far.txt and read the last line to get the tokens_so_far
        # load args from somewhere?
        # if args.folder_name is not set:
        #     folder_name = time.strftime("%Y%m%d-%H%M%S") + "_" + args.name
        #     args.folder_name = folder_name
        #     path = f"/home/mila/k/kusha.sareen/scratch/genPPO/evals/{args.folder_name}"
        #     os.makedirs(path, exist_ok=True)

        # reward_model.token_count = 0
        start_sample = 0
        print("Starting from sample: ", start_sample)
        asyncio.run(run_inference_sync(llm, reward_model, sampling_params, dataset, args, start_sample=start_sample))


async def run_inference(llm, reward_model, sampling_params, dataset, tokenizer, args):
    start = time.time()
    
    tasks, node_generator = collect_tasks(args, dataset, llm, reward_model, sampling_params, tokenizer)
    all_top_nodes = [await task for task in tasks]

    time_taken = time.time() - start
    total_tokens = node_generator.token_count + reward_model.token_count
    all_preds, all_different_scores = await parse_top_nodes(args, all_top_nodes, reward_model)

    # print("All preds: ", all_preds)
    log_everything(all_preds, all_different_scores, time_taken, all_top_nodes, total_tokens, args)
    run_evals(all_preds, all_different_scores, time_taken, total_tokens, args)


def collect_tasks(args, dataset, llm, reward_model, sampling_params, tokenizer):
    tasks = []
    num_samples = get_num_samples(args, dataset)
    _, node_generator = get_search_tree_and_generator(None, llm, reward_model, sampling_params, tokenizer, args)
    
    for i in range(num_samples):
        question = get_question(dataset, i, args)
        prompt = get_prompt(question, args)
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {"sum": 0, "min": 0, "last": 0} if args.log_all_scores else {args.aggregator: 0}) # 0 = 1 for logprobs
        tree, _ = get_search_tree_and_generator(root, llm, reward_model, sampling_params, tokenizer, args)
        tasks.append(asyncio.create_task(get_task(tree, root, node_generator, args)))
        gc.collect()

    return tasks, node_generator

async def run_inference_sync(llm, reward_model, sampling_params, dataset, args, start_sample = 0):
    start = time.time()
    
    num_samples = get_num_samples(args, dataset)
    all_top_nodes = []

    for i in range(start_sample, num_samples):
        question = get_question(dataset, i, args)
        prompt = get_prompt(question, args)
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {"sum": 0, "min": 0, "last": 0} if args.log_all_scores else {args.aggregator: 0}) # 0 = 1 for logprobs
        
        tree, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        top_nodes = await get_task(tree, root, node_generator, args)
        all_top_nodes.append(top_nodes)
        current_tokens = node_generator.token_count + reward_model.token_count
        log_solution(i, top_nodes, num_samples, current_tokens, args)
        gc.collect()

    
    time_taken = time.time() - start
    total_tokens = node_generator.token_count + reward_model.token_count
    log_args(args)
    all_preds, all_different_scores = await parse_top_nodes(args, all_top_nodes, reward_model)
    run_evals(all_preds, all_different_scores, time_taken, total_tokens, args)


if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
