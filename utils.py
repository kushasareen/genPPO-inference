from search_algorithms.beam_search import BeamSearchTree
from search_algorithms.best_of_n import BestOfNTree
from search_algorithms.rebase import RebaseTree
from generator import NodeGenerator, AsyncNodeGenerator
from reward_model import GenVinePPOVerifier, LLMAsAJudge, MathSphereRewardModel
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from datasets import Dataset, load_dataset
import asyncio
from omegaconf import DictConfig
import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import os
from reward_model import PPOVerifier
import numpy as np
import pickle

def load_inference_dataset(args):
    if 'MATH_250_test'.lower() in args.input_path.lower():
        dataset = load_dataset("nishadsinghi/MATH_250_test")['test']
    else:
        dataset = Dataset.load_from_disk(args.input_path)
    return dataset

def get_search_tree_and_generator(root , llm, reward_model, sampling_params, args):
    """
    Get the search tree and node generator based on the search algorithm.
    """

    if args.use_async:
        generator_type = AsyncNodeGenerator
    else:
        generator_type = NodeGenerator

    if args.search_algorithm == "beamsearch":
        tree = BeamSearchTree(root=root, beam_size=args.beam_size, beam_width=args.beam_width, top_k=args.top_k, use_advantage=args.use_advantage)
        generator = generator_type(llm, reward_model, num_children=args.beam_width, sampling_params=sampling_params, args=args)
    elif args.search_algorithm == "bestofn":
        tree =  BestOfNTree(root=root, n=args.n, top_k = args.top_k)
        generator = generator_type(llm, reward_model, num_children=1, sampling_params=sampling_params, args=args)
    elif args.search_algorithm == "rebase":
        tree = RebaseTree(root=root, expansion_temp=args.expansion_temp, top_k=args.top_k, use_advantage=args.use_advantage)
        generator = generator_type(llm, reward_model, num_children=None, sampling_params=sampling_params, args=args)
    else:
        raise ValueError(f"Search algorithm not implemented: {args.search_algorithm}")
    
    return tree, generator

def get_llm(model_name, args):
    gpu_memory_utilization = 0.4 if args.name == "rebase_rm" else 0.99
    max_model_len = 2048 if args.dataset == "gsm8k" else 2048
    dtype = 'bfloat16' if "qwen" in args.policy_model.lower() else 'float16'
    print("Dtype:", dtype)
    print("GPU Usage and Max Model Len:", gpu_memory_utilization, max_model_len)
    if args.use_async: 
        if args.dataset == "gsm8k":
            # llm = AsyncLLMEngine.from_engine_args(
            # AsyncEngineArgs(
            #     model=model_name,
            #     dtype='float16',
            #     enforce_eager=True,
            #     download_dir= args.download_dir,
            #     gpu_memory_utilization=gpu_memory_utilization,
            #     swap_space=3,
            #     max_model_len=max_model_len,
            #     kv_cache_dtype="fp8_e5m2",
            #     tensor_parallel_size=1, # needs to be more than 1 for tensor parallelism
            #     disable_log_requests=True
            #     )
            # )
            llm = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=model_name,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    trust_remote_code=True,
                    dtype='float16',
                    )
            )
        elif args.dataset == "math":
            llm = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=model_name,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    trust_remote_code=True,
                    dtype='float16',
                    seed = args.seed
                    )
            )

    else:
        llm = LLM(model=model_name,
                dtype='float16',
                max_model_len=2048,
                tensor_parallel_size=1, 
                download_dir = args.download_dir, 
                gpu_memory_utilization=gpu_memory_utilization, 
                enforce_eager=True) # False?
        
    
    tokenizer = asyncio.run(llm.get_tokenizer())
    return llm, tokenizer

def load_model(model_name, args):
    llm, tokenizer = get_llm(model_name, args)
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    if args.llm_as_judge:
        pass
    elif args.dataset == "math":
        stop_words.append("\n\n") # double check with arian
    elif args.dataset == "gsm8k":
        stop_words.append("\n")
    else:
        raise ValueError(f"Dataset not implemented: {args.dataset}")
    
    sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=args.max_tokens, stop=stop_words)
    return llm, sampling_params, tokenizer


def generate_filename(config: DictConfig, separator: str = "_", extension: str = ".json") -> str:
    """
    Generate a filename based on the attributes and values in a DictConfig.

    Args:
        config (DictConfig): The configuration object.
        separator (str): Separator to use between attributes and values.
        extension (str): File extension for the generated filename.

    Returns:
        str: Generated filename.
    """
    def sanitize(value):
        # Remove characters that are not safe for filenames
        return re.sub(r'[^\w\-]', '', str(value))

    parts = []
    for key, value in config.items():
        sanitized_key = sanitize(key)
        sanitized_value = sanitize(value)
        parts.append(f"{sanitized_key}{separator}{sanitized_value}")

    filename = separator.join(parts) + extension
    return filename

def save_results(results, args):
    filename = time.strftime("%Y%m%d-%H%M%S") + "_" + args.name + '_k' + str(args.top_k)+ '_s' + str(args.seed)

    path = args.output_path + "/" + filename
    with open(path, 'w+') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {path}")

def save_estimates(results, args):
    filename = time.strftime("%Y%m%d-%H%M%S") + "_" + args.name + '_k' + str(args.top_k)+ '_s' + str(args.seed)
    path = args.output_path + "/" + filename
    with open(path, 'w+') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {path}")


def get_reward_model(args, reward_llm, tokenizer):
    if args.llm_as_judge:
        return LLMAsAJudge(args, reward_llm, tokenizer)
    if args.ppo:
        return PPOVerifier(args, reward_llm, tokenizer)
    elif args.name == "rebase_rm":
        return MathSphereRewardModel(args, tokenizer)
    else:
        return GenVinePPOVerifier(args, reward_llm, tokenizer)
    
def get_all_models(args):
    if not args.ppo:
        llm, sampling_params, tokenizer = load_model(args.policy_model, args)
        if args.reward_model==args.policy_model:
            reward_llm = llm
        else:
            reward_llm, _, _ = load_model(args.reward_model, args)

        reward_model = get_reward_model(args, reward_llm, tokenizer)

    else:
        llm, reward_llm, tokenizer = load_ppo_model(args.policy_model, args.download_dir)
        stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
        stop_words.append("\n")
        sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=args.max_tokens, stop=stop_words)
        reward_model = get_reward_model(args, reward_llm, tokenizer)
        
    return llm, sampling_params, reward_model


def load_ppo_model(path, download_dir):
    import sys
    sys.path.append(r"/home/mila/k/kusha.sareen/genPPO/genPPO/src") # a little hacky

    from treetune.models.pretrained_with_value_head import PreTrainedModelForValueNetwork
    import os 

    max_model_len = 2048
    gpu_memory_utilization = 0.4
    print("GPU Usage and Max Model Len:", gpu_memory_utilization, max_model_len)
    # local_model_path = get_local_model_path(hf_model_name, download_dir)
    # model_dir = os.path.expanduser(local_model_path)
    # print(os.path.isdir(model_dir))
    # breakpoint()
    assert os.path.isdir(path), f"Model path {path} does not exist"
    generator = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model=f"{path}/hf_pretrained",
        dtype='float16',
        enforce_eager=True,
        download_dir= download_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=3,
        max_model_len=max_model_len,
        kv_cache_dtype="fp8_e5m2",
        tensor_parallel_size=1, # needs to be more than 1 for tensor parallelism
        disable_log_requests=True
        )
    )
    pretrained_backbone_model = AutoModelForCausalLM.from_pretrained(f"{path}/hf_pretrained", cache_dir="/network/scratch/k/kusha.sareen/cache")
    critic = PreTrainedModelForValueNetwork(pretrained_backbone_model)
    state_dict = torch.load(f"{path}/critic/hf_pretrained/pytorch_model.bin")
    print("state dict keys", state_dict.keys())
    # breakpoint()
    critic.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(f"{path}/hf_pretrained")
    return generator, critic, tokenizer

def download_ppo_model(hf_model_name, download_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(hf_model_name, cache_dir=download_dir)

def get_local_model_path(hf_model_name, download_dir):
    # should convert ReasoningMila/name to models--ReasoningMila--name
    return download_dir + "/models--" + hf_model_name.replace("/", "--") 

async def get_ppo_avg_orm_score(top_nodes, reward_model, different_scores):
    reward_model.mode = "mean_all"
    for idx, node in enumerate(top_nodes):
        prompt = node.state['text']
        solutions = ['']
        orm_score, _, _ = await reward_model(prompt, solutions) # returns a list of log probabilities
        different_scores[idx]["orm_avg"] = np.exp(orm_score[0])
    return different_scores

def get_question(dataset, i, args):
    sample = dataset[i]

    if args.dataset == "gsm8k":
        question = sample["question"]
    elif args.dataset == "math":
        question = sample["problem"]
    else:
        raise ValueError(f"Dataset not implemented: {args.dataset}")
    
    return question
    
def log_everything(all_preds, all_different_scores, time_taken, total_tokens, all_top_nodes, args):
    folder_name = time.strftime("%Y%m%d-%H%M%S") + "_" + args.name
    path = f"/home/mila/k/kusha.sareen/scratch/genPPO/evals/{folder_name}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/all_preds.pkl", 'wb') as f:
        pickle.dump(all_preds, f)
    
    with open(f"{path}/all_different_scores.pkl", 'wb') as f:
        pickle.dump(all_different_scores, f)

    with open(f"{path}/args.pkl", 'wb') as f:
        pickle.dump(args, f)

    with open(f"{path}/time_taken.pkl", 'wb') as f:
        pickle.dump(time_taken, f)

    with open(f"{path}/total_tokens.pkl", 'wb') as f:
        pickle.dump(total_tokens, f)

    with open(f"{path}/all_top_nodes.pkl", 'wb') as f:
        pickle.dump(all_top_nodes, f)

    print(f"Everything logged to: {path}")


def get_everything_from_logs(name):
    path = f"/home/mila/k/kusha.sareen/scratch/genPPO/evals/{name}"
    with open(f"{path}/all_preds.pkl", 'rb') as f:
        all_preds = pickle.load(f)
    
    with open(f"{path}/all_different_scores.pkl", 'rb') as f:
        all_different_scores = pickle.load(f)

    with open(f"{path}/args.pkl", 'rb') as f:
        args = pickle.load(f)

    with open(f"{path}/time_taken.pkl", 'rb') as f:
        time_taken = pickle.load(f)

    with open(f"{path}/total_tokens.pkl", 'rb') as f:
        total_tokens = pickle.load(f)

    with open(f"{path}/all_top_nodes.pkl", 'rb') as f:
        all_top_nodes = pickle.load(f)

    return all_preds, all_different_scores, time_taken, total_tokens, all_top_nodes, args

async def parse_top_nodes(args, all_top_nodes, reward_model):
    all_preds = []
    all_different_scores = []

    for top_nodes in all_top_nodes:
        predictions = [node.state['text'] for node in top_nodes]
        all_preds.append(predictions)
        different_scores = [{k: np.exp(v) for k, v in node.all_scores.items()} for node in top_nodes]
        if args.orm_avg:
            different_scores = await get_ppo_avg_orm_score(top_nodes, reward_model, different_scores)
        all_different_scores.append(different_scores)

    return all_preds, all_different_scores

if __name__ == "__main__":
    # from transformers import AutoModel
    # load_ppo_model("ReasoningMila/ppo_gsm_7b_ckpt_iter_0015_epoch_2.00_step_0240", "~/scratch/k/kusha.sareen/cache", connector_path='snapshots/f2e98aed5a4eb22b964453d2f920cf29af1b61be/hf_pretrained')
    # download_ppo_model("ReasoningMila/ppo_gsm_7b_ckpt_iter_0015_epoch_2.00_step_0240", "/network/scratch/k/kusha.sareen/cache")
    # /home/mila/k/kusha.sareen/scratch/cache/models--ReasoningMila--ppo_gsm_7b_ckpt_iter_0015_epoch_2.00_step_0240/snapshots/f2e98aed5a4eb22b964453d2f920cf29af1b61be/hf_pretrained
    # path = "/home/mila/k/kusha.sareen/scratch/cache/models--ReasoningMila--ppo_gsm_7b_ckpt_iter_0015_epoch_2.00_step_0240/snapshots/f2e98aed5a4eb22b964453d2f920cf29af1b61be"
    # print(os.path.isdir(path))
    # model = AutoModelForCausalLM.from_pretrained(path, cache_dir="/network/scratch/k/kusha.sareen/cache")

    generator = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model="/home/mila/k/kusha.sareen/scratch/genPPO/ppo_gsm_7b_ckpt_iter_0015_epoch_2.00_step_0240",
            dtype='float16',
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            swap_space=3,
            max_model_len=2048,
            kv_cache_dtype="fp8_e5m2",
            tensor_parallel_size=1, # needs to be more than 1 for tensor parallelism
            disable_log_requests=True
            )
        )
    
    print(generator)