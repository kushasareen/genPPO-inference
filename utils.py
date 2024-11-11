from search_algorithms.beam_search import BeamSearchTree
from search_algorithms.best_of_n import BestOfNTree
from generator import NodeGenerator, AsyncNodeGenerator
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from datasets import Dataset
import asyncio


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
        tree = BeamSearchTree(root=root, beam_width=args.beam_width)
        generator = generator_type(llm, reward_model, args.beam_width, sampling_params)
    elif args.search_algorithm == "bestofn":
        tree =  BestOfNTree(root=root, n=args.n)
        generator = generator_type(llm, reward_model, num_children=1, sampling_params=sampling_params)
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
                download_dir = "/network/scratch/k/kusha.sareen/cache", 
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
    sampling_params = SamplingParams(temperature=1.0, max_tokens=512, stop=stop_words)
    return llm, sampling_params, stop_words, tokenizer
