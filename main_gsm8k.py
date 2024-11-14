import argparse
import gc
from reward_model import GenVinePPOVerifier
from tree import TreeNode
from verify_gsm8k import evaluate_predictions
import time
from utils import get_search_tree_and_generator, load_dataset, load_model
import asyncio
import hydra

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_dataset(args)
    llm, sampling_params, stop_tokens, tokenizer = load_model(args.policy_model, args)
    reward_model = GenVinePPOVerifier(args, llm, tokenizer)
    start = time.time()
    asyncio.run(run_inference(llm, reward_model, sampling_params, dataset, args))
    end = time.time()
    print("Time: ", end - start)


async def run_inference(llm, reward_model, sampling_params, dataset, args):
    all_gts = []
    all_preds = []
    all_top_results = []
    tasks = []

    for i in range(len(dataset)):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        all_gts.append(answer) 
        prompt = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n' # prompt should match training data format
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0) 
        tree, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        tasks.append(asyncio.create_task(tree.search(generate_children=node_generator, max_depth=args.max_depth)))
        gc.collect()

    all_top_nodes = [await task for task in tasks]

    for top_nodes in all_top_nodes:
        predictions = [node.state['text'] for node in top_nodes]
        all_preds.append(predictions)
        all_top_results.append(top_nodes[0])

    print("\n**** Evaluating ****")
    results = evaluate_predictions(all_preds, dataset)

    print("\n**** Results ****")
    print(results)

    print("Config")
    print(args)

if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
