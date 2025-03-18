import gc
from verify_gsm8k import evaluate_predictions, estimate_token_count_at_k, estimate_time_at_k
import time
from utils import load_inference_dataset, save_results, log_everything, get_everything_from_logs, parse_by_question_logs, parse_top_nodes
from omegaconf import OmegaConf
import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="rebase_math")
    args = parser.parse_args()
    run_evals_new(args)

def run_evals_new(args):
    all_preds, all_different_scores, time_taken, total_tokens, args = collect_results_from_logs(args)
    dataset = load_inference_dataset(args)

    print("\n**** Evaluating ****")
    results = evaluate_predictions(all_preds, dataset, all_different_scores, args)

    print("\n**** Results ****")

    results["total_tokens"] = estimate_token_count_at_k(all_preds, total_tokens, args.top_k)

    results["time"] = estimate_time_at_k(all_preds, time_taken, args.top_k)

    results["config"] = OmegaConf.to_container(args, resolve = True)

    print(results)
    save_results(results, args)


def run_evals(all_preds, all_different_scores, time_taken, total_tokens, args):
    dataset = load_inference_dataset(args)

    print("\n**** Evaluating ****")
    results = evaluate_predictions(all_preds, dataset, all_different_scores, args)

    print("\n**** Results ****")

    results["total_tokens"] = estimate_token_count_at_k(all_preds, total_tokens, args.top_k)

    results["time"] = estimate_time_at_k(all_preds, time_taken, args.top_k)

    results["config"] = OmegaConf.to_container(args, resolve = True)

    print(results)
    save_results(results, args)

def collect_results_from_logs(args):
    name = args.name
    if args.use_async:
        all_preds, all_different_scores, time_taken, total_tokens, _, args = get_everything_from_logs(args.folder_name)
    else:
        all_top_nodes = parse_by_question_logs(args)
        all_preds, all_different_scores = asyncio.run(parse_top_nodes(args, all_top_nodes))
        time_taken = 0
        total_tokens = 0 # load total_tokens from file here
        args = None # load args from somewhere here

    return all_preds, all_different_scores, time_taken, total_tokens, args

if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()
