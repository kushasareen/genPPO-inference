import argparse
import json
import os
from vllm import LLM, SamplingParams
import gc
from generator import NodeGenerator
from reward_model import MathSphereRewardModel
from search_algorithms.beam_search import RandomizedBeamSearch, BeamSearchTree
from tree import TreeNode
from verify_math import extract_answer, extract_answer_by_box, grade_answer
import time
from datasets import Dataset


def load_dataset(args):
    dataset = Dataset.load_from_disk(args.input_path)
    return dataset

def load_model(model_name, args):
    llm = LLM(model=model_name,
            dtype='float16',
            max_model_len=2048,
            tensor_parallel_size=1, 
            download_dir = "/network/scratch/k/kusha.sareen/cache",  ### change this to your directory
            gpu_memory_utilization=0.5, 
            enforce_eager=True) 
    tokenizer = llm.get_tokenizer()
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    stop_words.append("ки")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512, stop = stop_words) #stop_tokens=stop_tokens)
    return llm, sampling_params, stop_words

def main(args):  
    dataset = load_dataset(args)
 
    policy, sampling_params, stop_tokens = load_model(args.policy_model, args)
    tokenizer = policy.get_tokenizer()
    reward_model = MathSphereRewardModel(args, tokenizer)

    beam_size = 4
    max_depth = 15
    beam_width = 8

    all_gts = []
    all_preds = []

    start = time.time()
    for i in range(1):
        i = 0
        sample = dataset[i]
        question = sample['problem']
        answer = sample['answer']
        all_gts.append(answer)
        prompt = question
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0}, score = 0, parent = None, depth = 0) 
        node_generator = NodeGenerator(policy, reward_model, beam_size, sampling_params)
        tree = BeamSearchTree(root=root, beam_width=beam_width)
        top_nodes = tree.beam_search(generate_children=node_generator, max_depth=max_depth)
        predictions = [node.state['text'] for node in top_nodes]
        predictions = extract_answer(predictions[0])
        all_preds.append(predictions)
        gc.collect()
    end = time.time()

    for ans, pred_ans in zip(all_gts, all_preds):
        print(f"Gold answer: {ans} | Pred answer: {pred_ans}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = '/network/scratch/k/khang.ngo/gen_vineppo/data/math/test')
    parser.add_argument("--policy_model", type = str, default = 'peiyi9979/math-shepherd-mistral-7b-rl')
    parser.add_argument("--reward_model", type = str, default = 'peiyi9979/math-shepherd-mistral-7b-prm')
    parser.add_argument('--device', default="cuda")

    args = parser.parse_args()

    try:
        main(args)
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()