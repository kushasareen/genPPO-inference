from search_algorithms.beam_search import BeamSearchTree
from search_algorithms.best_of_n import BestOfNTree
from search_algorithms.rebase import RebaseTree
from generator import NodeGenerator, AsyncNodeGenerator
from reward_model import GenVinePPOVerifier, LLMAsAJudge
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from datasets import Dataset
import asyncio
from omegaconf import DictConfig
import re
import json
import time

def load_dataset(args):
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
        tree = BeamSearchTree(root=root, beam_size=args.beam_size, beam_width=args.beam_width, top_k=args.top_k)
        generator = generator_type(llm, reward_model, num_children=args.beam_width, sampling_params=sampling_params, args=args)
    elif args.search_algorithm == "bestofn":
        tree =  BestOfNTree(root=root, n=args.n, top_k = args.top_k)
        generator = generator_type(llm, reward_model, num_children=1, sampling_params=sampling_params, args=args)
    elif args.search_algorithm == "rebase":
        tree = RebaseTree(root=root, expansion_temp=args.expansion_temp, top_k=args.top_k)
        generator = generator_type(llm, reward_model, num_children=None, sampling_params=sampling_params, args=args)
    else:
        raise ValueError(f"Search algorithm not implemented: {args.search_algorithm}")
    
    return tree, generator

def get_llm(model_name, args):
    if args.use_async: 
        llm = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_name,
            dtype='float16',
            enforce_eager=True,
            download_dir= args.download_dir,
            gpu_memory_utilization=0.99,
            swap_space=3,
            max_model_len=2048,
            kv_cache_dtype="fp8_e5m2",
            tensor_parallel_size=1, # needs to be more than 1 for tensor parallelism
            disable_log_requests=True
            )
        )

    else:
        llm = LLM(model=model_name,
                dtype='float16',
                max_model_len=2048,
                tensor_parallel_size=1, 
                download_dir = args.download_dir, 
                gpu_memory_utilization=0.5, 
                enforce_eager=True) # False?
        
    
    tokenizer = asyncio.run(llm.get_tokenizer())
    return llm, tokenizer

def load_dataset(args):
    dataset = Dataset.load_from_disk(args.input_path)
    return dataset

def load_model(model_name, args):
    llm, tokenizer = get_llm(model_name, args)
    stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
    stop_words.append("\n")
    sampling_params = SamplingParams(temperature=args.generation_temp, max_tokens=args.max_tokens, stop=stop_words)
    return llm, sampling_params, stop_words, tokenizer


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
    filename = time.strftime("%Y%m%d-%H%M%S") + "_" + args.name
    path = args.output_path + "/" + filename
    with open(path, 'w+') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {path}")

def save_estimates(results, args):
    filename = "mc_estimates_"+ time.strftime("%Y%m%d-%H%M%S") + "_" + args.name
    path = args.output_path + "/" + filename
    with open(path, 'w+') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {path}")


def get_reward_model(args):
    if args.llm_as_judge:
        return LLMAsAJudge
    else:
        return GenVinePPOVerifier