import argparse
import gc
from reward_model import GenVinePPOVerifier
from tree import TreeNode
from verify_gsm8k import evaluate_predictions
import time
from utils import get_search_tree_and_generator, load_dataset, load_model, save_results
import asyncio
import hydra
import numpy as np

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
    start = time.time()
    asyncio.run(run_inference(llm, reward_model, sampling_params, dataset, args))
    end = time.time()
    print("Time: ", end - start)


async def run_inference(llm, reward_model, sampling_params, dataset, args):
    all_gts = []
    all_preds = []
    all_top_results = []
    tasks = []
    all_different_scores = []

    for i in range(len(dataset)):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        all_gts.append(answer) 
        prompt = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n' # prompt should match training data format
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0, all_scores = {"sum": 0, "min": 0, "last": 0} if args.log_all_scores else {args.aggregator: 0})
        tree, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        tasks.append(asyncio.create_task(tree.search(generate_children=node_generator, max_depth=args.max_depth)))
        gc.collect()

    all_top_nodes = [await task for task in tasks]

    for top_nodes in all_top_nodes:
        predictions = [node.state['text'] for node in top_nodes]
        all_preds.append(predictions)
        all_top_results.append(top_nodes[0])
        different_scores = [{k: np.exp(v) for k, v in node.all_scores.items()} for node in top_nodes]
        all_different_scores.append(different_scores)


    print("\n**** Evaluating ****")
    results = evaluate_predictions(all_preds, dataset, all_different_scores, args)

    print("\n**** Results ****")
    print(results)
    print("Total tokens generated: ", node_generator.token_count + reward_model.token_count)
    results["total_tokens"] = node_generator.token_count + reward_model.token_count
    results["config"] = args
    save_results(results, args)

    print("Config")
    print(args)


if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
