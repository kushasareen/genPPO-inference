import argparse
import gc
from reward_model import GenVinePPOVerifier
from mc_estimator import MonteCarloEstimator
from tree import TreeNode
from verify_gsm8k import evaluate_predictions, estimate_token_count_at_k, estimate_time_at_k
import time
from utils import get_search_tree_and_generator, load_dataset, load_model, save_results, get_reward_model
import asyncio
import hydra
import numpy as np
from omegaconf import OmegaConf

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_dataset(args)
    llm, sampling_params, stop_tokens, tokenizer = load_model(args.policy_model, args)
    if args.reward_model==args.policy_model:
        reward_llm = llm
    else:
        reward_llm, _, _, _ = load_model(args.reward_model, args)

    reward_model = GenVinePPOVerifier(args, reward_llm, tokenizer)
    estimator = MonteCarloEstimator(llm, args)
    asyncio.run(run_validation(llm, reward_model, estimator, sampling_params, dataset, args))


async def run_validation(llm, reward_model, estimator, sampling_params, dataset, args): # TODO: adapt this for validation

    # save data to a file
    # will write a script to generate the graphs later on
    # can log accuracy vs. step
    
    start = time.time()

    all_gts = []
    all_preds = []
    all_top_results = []
    tasks = []
    all_different_scores = []

    for i in range(len(dataset)):
    # for i in range(3):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        all_gts.append(answer) 
        prompt = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n' # prompt should match training data format
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {"sum": 0, "min": 0, "last": 0} if args.log_all_scores else {args.aggregator: 0}) # 0 = 1 for logprobs
        tree, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        # search should be best of n sampling
        tasks.append(asyncio.create_task(tree.search(generate_children=node_generator, max_depth=args.max_depth)))
        gc.collect()

    all_top_nodes = [await task for task in tasks]
    all_paths = []
    path_scores = []

    for top_nodes in all_top_nodes: # FIX!!!!!!
        predictions = [node.state['text'] for node in top_nodes]
        all_preds.append(predictions)
        all_top_results.append(top_nodes[0])
        different_scores = [{k: np.exp(v) for k, v in node.all_scores.items()} for node in top_nodes]
        all_different_scores.append(different_scores)
        for node in top_nodes:
            path = node.path()
            all_paths.append(path)
            path_scores.append([node.score for node in path])

        
    sampled_nodes = top_nodes # TODO: sample some nodes from the path
    
    tasks = []
    for idx, nodes in enumerate(sampled_nodes):
        sample = dataset[idx] # nodes for a given quesion
        for node in nodes:
            tasks.append(asyncio.create_task(estimator.estimate(node[0], sample))) 

    all_estimates = [await task for task in tasks]

    results = {}
    results["model output"] = None
    results["ground truth"] = all_estimates
    total_tokens = node_generator.token_count
    results["total_tokens"] = total_tokens
    end = time.time()
    results["time"] = start - end
    results["config"] = OmegaConf.to_container(args, resolve = True)

    print(results)
    save_results(results, args)


if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
