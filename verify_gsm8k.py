import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional
import itertools
import numpy as np
from itertools import combinations
import scipy.special as scsp
from verify_math import grade_answer_math, extract_answer_math
from math_grader import math_equal
from parser_math import extract_answer as extract_answer_math_parser

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)

def extract_gold_answer_from_text(text: str) -> str:
    return text.split("####")[1].strip()

def extract_predicted_answer_from_text(text:str) -> str:
    text = text.replace(",", "")
    pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
    if len(pred_answer) == 0:
        return None
    else:
        # Pick the last number
        pred_answer = pred_answer[-1].strip()
        return pred_answer

def verify_float(answer: str, output: str):
    gt = extract_gold_answer_from_text(answer)
    pred_answer = FIND_NUMBERS_REGEX.findall(output)
    gt = float(gt)
    pred_answer = float(pred_answer[-1].strip()) 
    if abs(gt) >= 1:
        result = math.isclose(pred_answer, gt, abs_tol=0.1)
    else:
        result = math.isclose(pred_answer, gt, rel_tol=0.1)
    return result

def grade_answer( 
        given_answer: Optional[str] = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        if given_answer is None:
            return False

        assert ground_truth is not None
        return (
            given_answer.strip().replace(",", "").lower()
            == ground_truth.strip().lower()
        )

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def estimate_verifier_at_n(grading_results: List[bool], probs: List[float], n, num_subsets: int = 100) -> float:
    # sort correctness by verifier scores
    grading_results = np.array(grading_results)
    probs = np.array(probs)
    answers_with_verifier_scores = np.array(list(zip(grading_results, probs)))
    verifier_at_n_res = []
    correctness_scores = np.array([
            int(x)
            for _, x in sorted(zip([x[1] for x in answers_with_verifier_scores], grading_results), reverse=True)
        ])
    for _ in range(num_subsets):
        subset_indices = np.random.choice(len(grading_results), size=n, replace=False)
        correctness_scores_subset = correctness_scores[subset_indices]
        if n <= len(grading_results):
            verifier_at_n_res.append(verifier_at_k(correctness_scores, n))

    return np.mean(verifier_at_n_res)

                
def verifier_at_k(scores, k):
    EPS = 1e-9
    N = len(scores)  # Total number of samples
    # Pick ith example and (k-1) examples after it.
    numerators = [scores[i] * scsp.binom(N - i - 1, k - 1) for i in range(N - k + 1)]
    denominator = scsp.binom(N, k)  # - scsp.binom(N - C, k)
    fracs = [n / (denominator + EPS) for n in numerators]
    return sum(fracs)

def weighted_majority_vote(answers: List[str], probs: List[float], grading_results, k, num_subsets) -> str:
    if k > len(grading_results):
        print("Warning: k is larger than the number of answers. Setting k to the number of answers.")
        print("k vs number of answers:", k, len(grading_results))
        k = len(grading_results)

    answers_np = np.array(answers)
    probs = np.array(probs)
    num_correct = 0
    for _ in range(num_subsets):
        subset_indices = np.random.choice(len(grading_results), size=k, replace=False)
        answers_subset = answers_np[subset_indices]
        probs_subset = probs[subset_indices]
        answer_dict = {}
        for ans, p in zip(answers_subset, probs_subset):
            if ans in answer_dict:
                answer_dict[ans] += p
            else:
                answer_dict[ans] = p
                
        weighted_majority_answer = max(answer_dict, key=answer_dict.get)
        weighted_majority_answer_index = answers.index(weighted_majority_answer)
        weighted_majority_answer_is_correct = grading_results[weighted_majority_answer_index]
        num_correct += weighted_majority_answer_is_correct

    return num_correct / num_subsets

def majority_vote(answers: List[str], grading_results, k, num_subsets) -> str:
    if k > len(grading_results):
        print("Warning: k is larger than the number of answers. Setting k to the number of answers.")
        k = len(grading_results)

    answers_np = np.array(answers)
    num_correct = 0
    for _ in range(num_subsets):
        subset_indices = np.random.choice(len(grading_results), size=k, replace=False)
        answers_subset = answers_np[subset_indices]
        majority_answer, _ = Counter(answers_subset).most_common(n=1)[0]
        majority_answer_index = answers.index(majority_answer)
        majority_answer_is_correct = grading_results[majority_answer_index]
        num_correct += majority_answer_is_correct

    return num_correct / num_subsets

def powers_of_2_less_than(n):
    """Return a list of all powers of 2 less than n."""
    return [2 ** i for i in range(int(math.log2(n)) + 1)]

def estimate_token_count_at_k(predictions, token_count, top_k):
    min_solutions = min([len(sol) for sol in predictions]) 
    ks = powers_of_2_less_than(min_solutions-1)
    return {f"token_count@{k}":  (k/top_k)*token_count for k in ks}

def estimate_time_at_k(predictions, time, top_k):
    min_solutions = min([len(sol) for sol in predictions]) 
    ks = powers_of_2_less_than(min_solutions-1)
    return {f"time@{k}":  (k/top_k)*time for k in ks}

def evaluate_predictions(predictions: List[List[str]] = None, references : Any = None, all_scores = None, args = None) -> Dict[str, float]:
    once_hit_acc = []
    correct_frac = []
    unique_answer_count = []
    none_answer_extracted = []
    all_grading_results = []
    test_case_count = len(predictions)

    top1_acc = []
    majority_vote_acc = {}
    best_of_n_acc = {aggregator: {} for aggregator in all_scores[0][0].keys()}
    weighted_majority_vote_acc = {aggregator: {} for aggregator in all_scores[0][0].keys()}

    ### filter those that only have 1 solution ###
    predictions = list(filter(lambda sol: len(sol) > 1, predictions))
    min_solutions = min([len(sol) for sol in predictions])
    print(f"Min solutions: {min_solutions}")
    print([len(sol) for sol in predictions])
    ks = powers_of_2_less_than(min_solutions-1)
    ns = powers_of_2_less_than(min_solutions-1)

    ks.append(min_solutions-1)
    ns.append(min_solutions-1)


    for idx, (solution_candidates, ref) in enumerate(zip(predictions, references)):
        if args.dataset == "gsm8k":
            gold_answer = extract_gold_answer_from_text(ref["answer"])
            
            assert len(solution_candidates) > 0
            answer_candidates = [
                extract_predicted_answer_from_text(sol)
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )

            grading_results = [
                grade_answer(given_answer=ans, ground_truth=gold_answer, item=ref)
                for ans in answer_candidates
            ]
        elif args.dataset == "math":
            gold_answer = ref["answer"]
            answer_candidates = [
                extract_answer_math_parser(sol, data_name = "math")
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )
            grading_results = [
                grade_answer_math(given_answer=ans, ground_truth=gold_answer)
                for ans in answer_candidates
            ]
        else:
            raise ValueError("Unknown dataset")

        top1 = grading_results[0]
        
        top1_acc.append(top1)

        once_hit_acc.append(float(any(grading_results)))
        correct_frac.append(sum(grading_results) / len(grading_results))

        answer_candidates = [
            tuple(ans) if isinstance(ans, list) else ans
            for ans in answer_candidates
        ]

        majority_vote_acc[idx] = {
            k: majority_vote(answer_candidates, grading_results, k, num_subsets=100) for k in ks
        }

        for aggregator in all_scores[0][0].keys():
            prob = [all_scores[idx][i][aggregator] for i in range(len(all_scores[idx]))] # get the verifier scores for this problem and aggregator
            weighted_majority_vote_acc[aggregator][idx] = {
                k: weighted_majority_vote(answer_candidates, prob, grading_results, k, num_subsets=100) for k in ks
            }
            best_of_n_acc[aggregator][idx] = {
                n: estimate_verifier_at_n(grading_results, prob, n) for n in ns
            }


        unique_answer_count.append(len(set(answer_candidates)))

        all_grading_results.append(grading_results)

        try:
            topk_index = grading_results.index(True)
        except:
            topk_index = 0 ### no correct answer
        
        print(f"Test case: {idx} | Golden answer: {gold_answer} | Predicted answer: {answer_candidates[topk_index]}")

    once_hit = sum(once_hit_acc) / len(once_hit_acc)
    correct_frac = sum(correct_frac) / len(correct_frac)
    top1_acc = sum(top1_acc) / len(top1_acc)
    n_total = np.array([len(gr) for gr in all_grading_results])
    n_correct = np.array([sum(gr) for gr in all_grading_results])

    pass_at_k = {f"pass@{k}": estimate_pass_at_k(n_total, n_correct, k).mean() for k in ks if (n_total >= k).all()}
    
    # keys of output should be (aggregator, n) or (aggregator, k)
    # keys of input are (aggregator, problem_index, n) or (aggregator, problem_index, k), we need to average over problem_index
    overall_best_of_n_acc = {aggregator: {n: np.mean([best_of_n_acc[aggregator][i][n] for i in range(test_case_count)]) for n in ns} for aggregator in best_of_n_acc.keys()}
    overall_weighted_majority_vote_acc = {aggregator: {k: np.mean([weighted_majority_vote_acc[aggregator][i][k] for i in range(test_case_count)]) for k in ks} for aggregator in weighted_majority_vote_acc.keys()}

    overall_majority_vote_acc = {k: np.mean([majority_vote_acc[i][k] for i in range(test_case_count)]) for k in ks}
    base_results = {
        "top 1" : top1_acc,
        "once_hit": once_hit,
        "exact_match": once_hit,  # for backwards compatibility
        "correct_frac": correct_frac,
        "exact_match_frac": correct_frac,  # for backwards compatibility
        "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
        "none_answer_extracted_frac_per_problem": (
            sum(none_answer_extracted) / len(none_answer_extracted)
        ),}
    
    results_dict = {"base_results": base_results, 
                    "pass_at_k": pass_at_k, 
                    "best_of_n": overall_best_of_n_acc,
                    "majority_vote": overall_majority_vote_acc,
                    "weighted_majority_vote": overall_weighted_majority_vote_acc}        
    return results_dict


if __name__ == "__main__":
    # code for testing the evaluation function

    predictions = [
        ["42", "42", "42", "42"],  # Sample 1 predictions
        ["56", "32", "78", "89"],  # Sample 2 predictions
        ["12", "48", "92", "42"],  # Sample 3 predictions
        ["19", "35", "72", "29"],  # Sample 4 predictions
        ["87", "23", "55", "62"]  # Sample 5 predictions
    ]


    references = [
        {"answer": "Question: What is the answer? #### 42"},  # Sample 1 reference
        {"answer": "Question: What is the answer? #### 78"},  # Sample 2 reference
        {"answer": "Question: What is the answer? #### 48"},  # Sample 3 reference
        {"answer": "Question: What is the answer? #### 19"},  # Sample 4 reference
        {"answer": "Question: What is the answer? #### 55"}   # Sample 5 reference
    ]


    probabilities = [
        [0.4825235812382092, 0.3162943249438949, 0.201182093817896, 0.173753492005883],  # Sample 1 probabilities
        [0.3589722218924645, 0.2517771033942347, 0.21549718270741785, 0.173753492005883],  # Sample 2 probabilities
        [0.3292051670046432, 0.3317028361367011, 0.33909299685865575, 0.173753492005883],  # Sample 3 probabilities
        [0.24569759434675713, 0.23688577745273644, 0.2674708700646458, 0.2509457581358608],  # Sample 4 probabilities
        [0.12712653814246235, 0.2290187951174117, 0.3320874570593452, 0.1895792065155954]  # Sample 5 probabilities
    ]

    all_scores = [ 
        [{"sum": 0.4825235812382092, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.3162943249438949, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.201182093817896, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.173753492005883, "min": 0.173753492005883, "last": 0.173753492005883}],  # Sample 1 scores
        [{"sum": 0.3589722218924645, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.2517771033942347, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.21549718270741785, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.173753492005883, "min": 0.173753492005883, "last": 0.173753492005883}],  # Sample 2 scores
        [{"sum": 0.3589722218924645, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.2517771033942347, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.21549718270741785, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.173753492005883, "min": 0.173753492005883, "last": 0.173753492005883}],  # Sample 2 scores
        [{"sum": 0.3589722218924645, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.2517771033942347, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.21549718270741785, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.173753492005883, "min": 0.173753492005883, "last": 0.173753492005883}],  # Sample 2 scores
        [{"sum": 0.3589722218924645, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.2517771033942347, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.21549718270741785, "min": 0.173753492005883, "last": 0.173753492005883}, {"sum": 0.173753492005883, "min": 0.173753492005883, "last": 0.173753492005883}],  # Sample 2 scores
    ]
    class args:
        log_all_scores = False
    results = evaluate_predictions(predictions, references, all_scores, args)
    print(results)