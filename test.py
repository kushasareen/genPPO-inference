import argparse
import json
import os
from vllm import LLM, SamplingParams
import gc
from generator import NodeGenerator
from reward_model import MathSphereRewardModel
from search_algorithms.beam_search import RandomizedBeamSearch, BeamSearchTree
from tree import TreeNode
from verify_gsm8k import verify_float, evaluate_predictions
import time
from prompts import single_reflection_prompt_en, MetaMathPrompt

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_prompts(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        if 'problem' in test:
            prompts.append(test["problem"])
        else:
            prompts.append(test['question'])
    return prompts, test_cases

def load_model(model_name = 'meta-math/MetaMath-Mistral-7B', args = None):
    llm = LLM(model=model_name,
            dtype='float16',
            max_model_len=2048,
            tensor_parallel_size=1, 
            download_dir = "/network/scratch/k/khang.ngo/cache", 
            gpu_memory_utilization=0.5, 
            enforce_eager=True)
            #token = "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") 
    tokenizer = llm.get_tokenizer()
    yes_token_id = tokenizer.convert_tokens_to_ids('Yes')
    no_token_id = tokenizer.convert_tokens_to_ids('No')
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    stop_words.append("\n")
    sampling_params_1 = SamplingParams(temperature=0.0, max_tokens=512, stop= stop_words) #stop = stop_words) #stop_tokens=stop_tokens)
    sampling_params_2 = SamplingParams(temperature=0.0, max_tokens=512, logprobs=True) #stop = stop_words)
    return llm, sampling_params_1, sampling_params_2, yes_token_id, no_token_id

def run_inference(model, sampling_params, prompt):
    """Run inference on the given model with a prompt."""     
    responses = model.generate(prompt, sampling_params)
    return responses

def extract_answer(text:str):
    return text.split("####")[1].strip()

from datasets import Dataset
def load_dataset(args):
    dataset = Dataset.load_from_disk(args.input_path)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_path", type = str, default = '/network/scratch/k/khang.ngo/gen_vineppo/datasets/gsm8k/test.jsonl')
    parser.add_argument("--input_path", type = str, default = '/network/scratch/k/khang.ngo/gen_vineppo/data/gsm8k/test')
    parser.add_argument("--policy_model", type = str, default = 'peiyi9979/math-shepherd-mistral-7b-rl')
    parser.add_argument("--reward_model", type = str, default = 'peiyi9979/math-shepherd-mistral-7b-prm')
    parser.add_argument('--device', default="cuda")

    args = parser.parse_args()
    #args.policy_model = "deepseek-ai/deepseek-math-7b-rl"
    args.policy_model = "ReasoningMila/genppo_init_ckpt"
    # yes_token_id = ''

    try:
        policy, sampling_params_1, sampling_params_2, yes_token_id, no_token_id = load_model(model_name = args.policy_model)
        dataset = load_dataset(args)
        sample = dataset[25]
        question = sample['question']
        answer = sample['answer']
        #prompt = "Answer the following question: " + question + "\nLet's think step by step!"
        prompt = "Problem:\n" + question + '\nSolution:\n'

        all_steps = []
        verification_question = "\nIs the solution likely to result in the correct answer (Yes/No)?"
    
        for step in range(20):
            responses = run_inference(policy, sampling_params_1, prompt)
            solution = responses[0].outputs[0].text 
            verification_prompt = prompt + solution + verification_question
            verified_responses = run_inference(policy, sampling_params_2, verification_prompt) 
            first_output = verified_responses[0].outputs[0].logprobs[0]
            token = first_output[yes_token_id].decoded_token
            logprob = first_output[yes_token_id].logprob
            #print(first_token)
            print(token)
            print(logprob)
            break
            yes_token = verified_responses[0].outputs[0].text[:3]
            prompt = prompt  + solution + '\n' #' Is this step correct (Yes/No)? ' + yes_token + '.\n'
            if "####" in solution:
                pred_answer = extract_answer(solution)
                break
            print(f"Step: {step}", solution)
            print('Verification: ', verified_responses[0].outputs[0].text)
            print("========")
            # print(prompt)
            print("********")
        print(prompt)
        print(f"Golden answer: {answer} | Predicted answer: {pred_answer}")
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
