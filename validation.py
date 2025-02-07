import random
from mc_estimator import MCEstimator
import gc
import argparse
from utils import get_everything_from_logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="rebase_math")
    args = parser.parse_args()
    name = args.name
    all_preds, all_different_scores, time_taken, total_tokens, _, args = get_everything_from_logs(name)

def run_validation(llm, all_top_nodes, sampling_params, dataset, num_mc_est, num_samples):
    results = {}  # Store "problem_id", "solution_id" → {"logprob", "mc_value", "depth"}

    validation_dataset = []

    # Collect parent nodes for validation
    for problem_id, top_nodes in enumerate(all_top_nodes):
        for node in top_nodes:
            while node.parent is not None:  # Traverse upwards to collect parents
                validation_dataset.append(node.parent)
                node = node.parent  

    # Subsample a random subset of nodes for validation
    num_samples = min(num_samples, len(validation_dataset))
    sampled_nodes = random.sample(validation_dataset, num_samples)

    for node in sampled_nodes:
        problem_id = node.problem_id  # Assuming node has problem_id attribute
        solution_id = node.solution_id  # Assuming unique ID for solutions

        # Run LLM inference to generate multiple solutions
        generated_solutions = llm.infer(node.state["text"], num_samples=num_mc_est) # TODO: Add vLLM code here

        # Compute fraction of correct solutions
        correct_count = sum(1 for sol in generated_solutions if is_correct(sol))
        mc_value = correct_count / num_mc_est

        # Store results
        results[(problem_id, solution_id)] = {
            "logprob": node.state["logprob"],
            "mc_value": mc_value,
            "depth": node.depth,
        }

    return results

def is_correct(solution): # TODO: Implement this function
    """Placeholder function to check if a solution is correct."""
    return "boxed" in solution  # Example heuristic

def save_results(results, filename):
    """Save validation results to a file."""
    path = f"validation/{filename}.csv"
    with open(path, "w") as f:
        for (problem_id, solution_id), result in results.items():
            f.write(f"{problem_id},{solution_id},{result['logprob']},{result['mc_value']},{result['depth']}\n")

def load_results(filename):
    """Load validation results from a file."""
    path = f"validation/{filename}.csv"
    results = {}
    with open(path, "r") as f:
        for line in f:
            problem_id, solution_id, logprob, mc_value, depth = line.strip().split(",")
            results[(int(problem_id), int(solution_id))] = {
                "logprob": float(logprob),
                "mc_value": float(mc_value),
                "depth": int(depth),
            }

    return results


if __name__ == "__main__":

    try:
        main()
        gc.collect()
    except ValueError as e:
        print(e)
        gc.collect()