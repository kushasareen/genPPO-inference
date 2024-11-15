import argparse
import os
import gc
from generator import NodeGenerator
from reward_model import GenVinePPOVerifier
from search_algorithms.beam_search import BeamSearchTree
from tree import TreeNode
from verify_gsm8k import evaluate_predictions
import time
from verify_gsm8k import extract_gold_answer_from_text
from utils import get_search_tree_and_generator, load_dataset, load_model
import asyncio

def main(args):  
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
        print(f"Test case: ", i)
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        all_gts.append(answer) 
        prompt = "Problem:\n" + question + '\nSolution:\n'
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = '/network/scratch/k/kusha.sareen/genPPO/data/gsm8k/test')
    parser.add_argument("--policy_model", type = str, default = 'ReasoningMila/genppo_init_ckpt')
    parser.add_argument("--reward_model", type = str, default = 'ReasoningMila/genppo_init_ckpt')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--search_algorithm', type=str, default='beamsearch')
    parser.add_argument('--use_async', type=bool, default=True)    

    args = parser.parse_args()

    try:
        main(args)
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
