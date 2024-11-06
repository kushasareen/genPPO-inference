import argparse
import os
from vllm import LLM, SamplingParams
import gc
from generator import NodeGenerator
from reward_model import GenVinePPOVerifier
from search_algorithms.beam_search import BeamSearchTree
from tree import TreeNode
from verify_gsm8k import evaluate_predictions
import time
from datasets import Dataset
from verify_gsm8k import extract_gold_answer_from_text

def load_dataset(args):
    dataset = Dataset.load_from_disk(args.input_path)
    return dataset

def load_model(model_name, args):
    llm = LLM(model=model_name,
            dtype='float16',
            max_model_len=2048,
            tensor_parallel_size=1, 
            download_dir = "/network/scratch/k/khang.ngo/cache", 
            gpu_memory_utilization=0.5, 
            enforce_eager=True) 
    tokenizer = llm.get_tokenizer()
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    stop_words.append("\n")
    sampling_params = SamplingParams(temperature=1.0, max_tokens=512, stop=stop_words)
    return llm, sampling_params, stop_words

def main(args):  
    dataset = load_dataset(args)
    llm, sampling_params, stop_tokens = load_model(args.policy_model, args)
    reward_model = GenVinePPOVerifier(args, llm)

    beam_size = 4
    max_depth = 10
    beam_width = 4

    all_gts = []
    all_preds = []
    all_top_results = []

    start = time.time()
    for i in range(len(dataset)):
        print(f"Test case: ", i)
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        all_gts.append(answer) 
        prompt = "Problem:\n" + question + '\nSolution:\n'
        root = TreeNode(state = {'text' : prompt, 'logprob' : 0, 'token' : '', 'step_solution' : '', 'full_feedback' : ''}, 
                        score = 0, parent = None, depth = 0) 
        node_generator = NodeGenerator(llm, reward_model, beam_size, sampling_params)
        tree = BeamSearchTree(root=root, beam_width=beam_width)
        top_nodes = tree.beam_search(generate_children=node_generator, max_depth=max_depth)
        predictions = [node.state['text'] for node in top_nodes]
        all_preds.append(predictions)
        all_top_results.append(top_nodes[0])
        gc.collect()
    end = time.time()

    print("\n**** Evaluating ****")
    results = evaluate_predictions(all_preds, dataset)
    #print("*" * 10)

    print("\n**** Results ****")
    print(results)
    print("Time: ", end - start)

    # print("\n*** Predicted solutions ***")
    # for i, predictions in enumerate(all_preds):
    #     print(f"Test case: {i}")
    #     print(predictions[0]) 
    #     print("=======")

    # print("\n============")
    # for idx, node in enumerate(all_top_results):
    #     print(f"Test case: {idx} | Golden answer: {extract_gold_answer_from_text(all_gts[idx])}")
    #     for parent in node.path():
    #         print(f'Step solution: ',parent.state['step_solution']) 
    #         print(f'Verification: ', parent.state['token'])
    #         # print(f"Full Feedback: ", parent.state['full_feedback'])
    #         print(f'Depth: ', parent.depth) 
    #     print("********") 
    #print("*******")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = '/network/scratch/k/khang.ngo/gen_vineppo/datasets/prm800k/prm800k/math_splits/test.jsonl')
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
