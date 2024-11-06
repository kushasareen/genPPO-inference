import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

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

def evaluate_predictions(predictions: List[List[str]] = None, references : Any = None):
    once_hit_acc = []
    correct_frac = []
    majority_vote_acc = []
    unique_answer_count = []
    none_answer_extracted = []

    for idx, (solution_candidates, ref) in enumerate(zip(predictions, references)):
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
        once_hit_acc.append(float(any(grading_results)))
        correct_frac.append(sum(grading_results) / len(grading_results))

        answer_candidates = [
            tuple(ans) if isinstance(ans, list) else ans
            for ans in answer_candidates
        ]

        majority_answer, _ = Counter(answer_candidates).most_common(n=1)[0]
        assert len(answer_candidates) == len(grading_results)
        majority_answer_index = answer_candidates.index(majority_answer)
        majority_answer_is_correct = grading_results[majority_answer_index]
        majority_vote_acc.append(majority_answer_is_correct)

        unique_answer_count.append(len(set(answer_candidates)))

        print(f"Test case: {idx} | Golden answer: {gold_answer} | Predicted answer: {majority_answer}")

    once_hit = sum(once_hit_acc) / len(once_hit_acc)
    correct_frac = sum(correct_frac) / len(correct_frac)

    return {
        "once_hit": once_hit,
        "exact_match": once_hit,  # for backwards compatibility
        "correct_frac": correct_frac,
        "exact_match_frac": correct_frac,  # for backwards compatibility
        "majority_vote_acc": sum(majority_vote_acc) / len(majority_vote_acc),
        "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
        "none_answer_extracted_frac_per_problem": (
            sum(none_answer_extracted) / len(none_answer_extracted)
        ),
    }
