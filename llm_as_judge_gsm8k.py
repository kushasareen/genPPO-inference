import argparse
import gc
from reward_model import GenVinePPOVerifier, LLMAsAJudge
from tree import TreeNode
from verify_gsm8k import evaluate_predictions, estimate_token_count_at_k, estimate_time_at_k
import time
from utils import get_search_tree_and_generator, load_dataset, load_model, save_results, get_reward_model, get_llm
from vllm import SamplingParams
import asyncio
import hydra
import numpy as np
from omegaconf import OmegaConf

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_dataset(args)
    llm, tokenizer = get_llm(args.policy_model, args)
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>'] # \n no longer in stop words
    sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=args.max_tokens, stop=stop_words)
    assert args.reward_model==args.policy_model

    reward_model = LLMAsAJudge(args, llm, tokenizer)
    asyncio.run(run_inference(llm, reward_model, sampling_params, dataset, args))


async def run_inference(llm, reward_model, sampling_params, dataset, args):
    start = time.time()

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
                        score = 0, parent = None, depth = 0, all_scores = {args.aggregator: 0}) # 0 = 1 for logprobs
        
        _, node_generator = get_search_tree_and_generator(root, llm, reward_model, sampling_params, args)

        tasks.append(asyncio.create_task(node_generator(root, width=args.top_k)))
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

    total_tokens = node_generator.token_count + reward_model.token_count
    results["total_tokens"] = estimate_token_count_at_k(all_preds, total_tokens, args.top_k)

    end = time.time()
    results["time"] = estimate_time_at_k(all_preds, end - start, args.top_k)

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
