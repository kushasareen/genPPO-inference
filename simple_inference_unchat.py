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
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import time
import os
from reward_model import PPOVerifier
import numpy as np
from generator import run_async_inference
import uuid
import re

@hydra.main(version_base = None, config_path="configs", config_name="default")
def main(cfg):  
    args = cfg.search_algorithm
    print(args)
    dataset = load_inference_dataset(args)
    llm = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=args.policy_model,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    trust_remote_code=True,
                    dtype='float16',
                    seed = args.seed,
                    )
            )
    tokenizer = asyncio.run(llm.get_tokenizer())
    reward_llm = llm
    reward_model = GenVinePPOVerifier(args, reward_llm, tokenizer)
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    print("Stop words: ", stop_words)
    print("Max tokens: ", args.max_tokens)
    sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=args.max_tokens, stop=stop_words, include_stop_str_in_output=True)
    verification_sampling_params = SamplingParams(temperature=args.verification_temp, max_tokens=1, logprobs=20)
    asyncio.run(run_inference(llm, reward_model, sampling_params, verification_sampling_params, dataset, tokenizer, args))
    

async def run_inference(llm, reward_model, sampling_params, verification_sampling_params, dataset, tokenizer, args):
    start = time.time()

    num_samples = get_num_samples(args, dataset)
    all_preds = []
    all_different_scores = []

    yes_token_id = tokenizer.convert_tokens_to_ids('Yes')

    for i in range(num_samples):
        question = get_question(dataset, i, args)
        question_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + question + "\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant"
        prompts = [question_prompt] * args.top_k
        tasks = [asyncio.create_task(run_async_inference(llm, sampling_params, prompt, uuid.uuid4())) for prompt in prompts]
        responses = [await task for task in tasks]
        # responses_text = [remove_after_boxed(response.outputs[0].text) for response in responses]
        responses_text = [response.outputs[0].text for response in responses]
        all_preds.append(responses_text)
        scores = []
        for response in responses_text:
            scores.append({"last": 0})

        all_different_scores.append(scores)
        print(f"Problem {i + 1} of {num_samples} done")
        # breakpoint()
        gc.collect()

    time_taken = time.time() - start
    total_tokens = 0
    log_everything(all_preds, all_different_scores, time_taken, [], total_tokens, args)
    run_evals(all_preds, all_different_scores, time_taken, total_tokens, args)

if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
