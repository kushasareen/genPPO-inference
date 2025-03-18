import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc
import argparse
from utils import get_everything_from_logs, load_inference_dataset
from verify_math import grade_answer_math, extract_answer_math
from math_grader import math_equal
from parser_math import extract_answer as extract_answer_math_parser
import random
from tqdm import tqdm
import numpy as np
import os

def main(path):
    if not os.path.exists(f"./plotting/plots/confusion_matrices/{path}"):
        os.makedirs(f"./plotting/plots/confusion_matrices/{path}")

    all_preds, all_different_scores, time_taken, total_tokens, all_top_nodes, cfg = get_everything_from_logs(path)
    print(cfg)
    dataset = load_inference_dataset(cfg)
    # plot confusion matrix
    y_pred = []
    y_true = []
    for i in tqdm(range(len(all_preds))):
        for j in range(len(all_preds[i])):
            score = all_different_scores[i][j]["last"]
            ref = dataset[i]
            if "answer" in ref:
                gold_answer = ref["answer"]
            else:
                sol = ref["solution"]
                gold_answer = extract_answer_math_parser(sol, data_name = "math")

            sol = all_preds[i][j]
            ans = extract_answer_math_parser(sol, data_name = "math")
            if ans == "": # skip no answer in validation
                continue

            if score < 0.5:
                y_pred.append(0)
            else:
                y_pred.append(1)

            grading_result = grade_answer_math(given_answer=ans, ground_truth=gold_answer)
            y_true.append(grading_result)
            # if not grading_result: breakpoint()
            # if j % 32 == 0: breakpoint()

        
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='.3f')
    plt.savefig(f"./plotting/plots/confusion_matrices/{path}/confusion_matrix.png")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='.3f')
    plt.savefig(f"./plotting/plots/confusion_matrices/{path}/confusion_matrix_normalized.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="None")
    args = parser.parse_args()

    # paths = [
    #     "20250225-200230_bestofn_math_k512_qwen_ORM_1_aime",
    #     "20250225-192920_bestofn_math_k512_qwen_ORM_2_aime",
    #     "20250225-155936_bestofn_math_k512_qwen_ORM_4_aime",
    #     "20250225-191906_bestofn_math_k512_qwen_ORM_8_aime",
    #     "20250225-195242_bestofn_math_k512_qwen_ORM_16_aime"
    #     ]
    # paths = [
    #     "20250225-225903_bestofn_math_k512_qwen_ORM_4_math128",
    #     "20250226-100821_bestofn_math_k512_qwen_ORM_16_math128",
    # ]
    # paths = [
    #     "20250226-122121_bestofn_math_k512_qwen_ORM_8_math128",
    #     "20250226-123826_bestofn_math_k512_qwen_ORM_2_math128",
    #     "20250226-124414_bestofn_math_k512_qwen_ORM_1_math128"
    # ]

    # paths = [
    #     "20250226-170358_bestofn_math_k32_qwen_ORM_1_math128",
    #     "20250226-172103_bestofn_math_k32_clf_init_math128"
    # ]

    # paths = [
        # "20250309-111835_bestofn_grpo_k32_model_math128", # 0.1
        # "20250309-112837_bestofn_grpo_k32_model_math128", # 0
        # "20250309-114337_bestofn_grpo_k32_model_math128" # 0.5
        # "20250308-135718_bestofn_math_k32_sft_no_instr_1_math128",
        # "20250308-140147_bestofn_math_k32_clf_no_instr_1_math128"
        # "20250228-103112_bestofn_math_k64_sft_init_math128"
    # ]

    # paths = [
    #     "20250313-093848_bestofn_grpo_k32_model_math128", # 0
    #     "20250313-093948_bestofn_grpo_k32_model_math128", # 0.1_clf
    #     "20250313-094417_bestofn_grpo_k32_model_math128", # 0.1_sft
    # ]

    # paths = [
    #     "20250314-090443_bestofn_grpo_k64_model_math128",
    #     "20250314-090811_bestofn_grpo_k64_model_math128",
    #     "20250314-093356_bestofn_grpo_k64_model_math128"
    # ]

    # paths = [
    #     "grpo_scot_math_0.8_sft_5e-5_1742040760_42_bestofn_grpo_k64_math128",
    #     "grpo_scot_math_1_sft_5e-5_1742029900_42_bestofn_grpo_k64_math128"
    # ]

    # paths = [
    #     'grpo_scot_math_0.1_clf_5e-5_1741997318_42_bestofn_grpo_k64_math128',
    #     'grpo_scot_math_0.8_sft_5e-5_1742040760_42_bestofn_grpo_k64_math128'
    # ]

    if args.path is None:
        paths = [
            "grpo_scot_math_1_clf_1.5e-4__42_bestofn_grpo_k256_math128",
        ]
    else:
        paths = [args.path]
        
    for idx, path in enumerate(paths):
        print(f"Plotting confusion matrix for {path}")
        print(f"Path {idx+1} of {len(paths)}")
        print("=====================================")
        main(path)
    # main(args.path)
    gc.collect()