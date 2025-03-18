import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

rcParams.update({'font.size': 14})  # Increase base font size for the entire plot

def load_csvs(model_names):
    data = {}
    for model in model_names:
        path = f"/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}"
        csv_files = os.listdir(path)
        for file in csv_files:
            # Check if the file is a CSV file (and not a directory)
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and file.lower().endswith('.csv'):
                data[(model, file)] = pd.read_csv(file_path)

    return data

def get_linestyles(num_linestyles):
    assert num_linestyles <= 4, "Only 4 linestyles are supported"
    return ['solid', 'dashed', 'dotted', 'dashdot'][:num_linestyles]

def get_base_colors(method_names):
    print(method_names)
    assert len(method_names) <= 13, "Only 13 base colors are supported"

    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "violet",  # Violet
        "yellow",  # Yellow
        "black",  # Black
    ]

    return {method: colors[i] for i, method in enumerate(method_names)}

def plot_method(method_data, model, file, fig_maj, fig_bon, fig_wmaj, fig_pan, fig_tok, color_dict, suffix_to_plot, only_best_suffix=False):
    # Extract the metric names
    search_method = file.split(".")[0].split("_")[0]
        # Get unique base method names
    base_methods = sorted(set(m.split("-sum")[0].split("-min")[0].split("-last")[0].split("-orm_avg")[0] for m in method_data["metric"]))
    base_methods.remove("time")

    method_data["suffix"] = method_data["metric"].apply(lambda x: x.split("-")[-1] if "-" in x else "")
    
    # Define linestyle mapping
    linestyle_map = {"sum": "solid", "min": "dashed", "last": "dotted", "orm_avg": "dashdot"}
    
    # Define figure mapping for different plot types
    fig_map = {
        "majority_vote": fig_maj,
        "best_of_n": fig_bon,
        "weighted_majority_vote": fig_wmaj,
        "pass_at_k": fig_pan,
        "total_tokens": fig_tok,
    }

    # Assign colors to base methods
    for adv in [True, False]:
        if method_data[method_data["adv"] == adv].empty:
            continue

        model_and_search_name = "_".join([model, search_method])
        if adv:
            model_and_search_name += "_adv"

        color = color_dict[model_and_search_name]
        method_data_adv = method_data[method_data["adv"] == adv]

        for base_method in base_methods:
            sub_data = method_data_adv[method_data["metric"].str.startswith(base_method)]
            tok_data = method_data_adv[method_data["metric"].str.startswith("total_tokens")]
            # Determine which figure to use
            key = base_method
            fig = fig_map.get(key)
            if fig == None: breakpoint()

            ax = fig.gca()

            if sub_data["suffix"].nunique() == 1:
                # sort by true_k
                sub_data = sub_data.sort_values("true_k")
                tok_data = tok_data.sort_values("value")
                name = model_and_search_name
                if 'bestofn' in name:
                    name = name.replace('bestofn', 'sampling')

                if len(sub_data) != len(tok_data):
                    sub_data = sub_data[:len(tok_data)]

                ax.errorbar(
                    sub_data["k"],
                    # tok_data["value"],
                    sub_data["value"],
                    yerr=sub_data["std"],
                    label=f"{name}",
                    linestyle="solid",
                    color=color,
                    marker="o",
                    capsize=3
                )
                print(key, model_and_search_name)
                ax.legend()  # Ensure legend is added if any labels exist
                ax.set_title(key)
                # ax.set_xlabel("Tokens")
                ax.set_xlabel("k")
                ax.set_ylabel("Value")
                ax.set_xscale("log", base=2)
                min_tok = min(tok_data["value"])
                max_tok = max(tok_data["value"])
                # ax.set_xlim([min_tok//2, 2*max_tok])


                if key == "total_tokens":
                    ax.set_yscale("log", base=2)
                    ax.set_ylim([2**9, 2**18])
                ax.grid(True)
            else:
                if only_best_suffix:
                    best_suffix = None
                    best_row_idx = sub_data.idxmax()["value"]
                    best_row = sub_data.loc[best_row_idx]
                    best_suffix = best_row["suffix"]

                for suffix, linestyle in linestyle_map.items():
                    if suffix not in suffix_to_plot:
                        continue

                    if only_best_suffix and suffix != best_suffix:
                        continue

                    sub_sub_data = sub_data[sub_data["metric"].str.endswith(suffix)]
                    if sub_sub_data.empty:
                        print(f"Skipping {model_and_search_name} for {base_method} with suffix {suffix}")
                        continue

                    tok_data = method_data_adv[method_data["metric"].str.startswith("total_tokens")]
                    tok_data = tok_data.sort_values("value")
                    # sort by true_k
                    sub_sub_data = sub_sub_data.sort_values("k")
                    name = model_and_search_name
                    if 'bestofn' in name:
                        name = name.replace('bestofn', 'sampling')

                    if len(sub_sub_data) != len(tok_data):
                        sub_sub_data = sub_sub_data[:len(tok_data)]

                    ax.errorbar(
                        sub_sub_data["k"],
                        # tok_data["value"],
                        sub_sub_data["value"],
                        yerr=sub_sub_data["std"],
                        label=f"{name} ({suffix})",
                        linestyle=linestyle,
                        color=color,
                        marker="o",
                        capsize=3
                    )
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax.set_title(key)
                    ax.set_xlabel("k")
                    ax.set_ylabel("Value")
                    ax.set_xscale("log", base=2)
                    min_tok = min(tok_data["value"])
                    max_tok = max(tok_data["value"])
                    # ax.set_xlim([min_tok//2, 2*max_tok])
                    ax.grid(True)

    return fig_maj, fig_bon, fig_wmaj, fig_pan, fig_tok

def flatten_list(l):
    return [item for sublist in l for item in sublist]
    
def plot_all(data, model_names, save_path="plots", suffix_to_plot=["sum", "min", "last", "orm_avg"], best_suffix=False):
    if any("rebase" in filename for (_, filename) in data.keys()):
        model_and_search_names = [[f"{model}_rebase", f"{model}_rebase_adv", f"{model}_bestofn"] for model in model_names]
    else:
        model_and_search_names = [[f"{model}_bestofn"] for model in model_names]

    model_and_search_names = flatten_list(model_and_search_names)

    color_dict = get_base_colors(model_and_search_names)
    f_maj = plt.figure(figsize=(12, 6), dpi=300)
    f_bon = plt.figure(figsize=(12, 6), dpi=300)
    f_wmaj = plt.figure(figsize=(12, 6), dpi=300)
    f_pan = plt.figure(figsize=(12, 6), dpi=300)
    f_tok = plt.figure(figsize=(12, 6), dpi=300)

    for (model, file), model_data in data.items():
        fig_maj, fig_bon, fig_wmaj, fig_pan, fig_tok = plot_method(model_data, model, file, f_maj, f_bon, f_wmaj, f_pan, f_tok, color_dict, suffix_to_plot=suffix_to_plot, only_best_suffix=best_suffix)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f_maj.savefig(f"{save_path}/majority_vote.png", bbox_inches='tight')
    f_bon.savefig(f"{save_path}/best_of_n.png", bbox_inches='tight')
    f_wmaj.savefig(f"{save_path}/weighted_majority_vote.png", bbox_inches='tight')
    f_pan.savefig(f"{save_path}/pass_at_n.png", bbox_inches='tight')
    f_tok.savefig(f"{save_path}/tokens.png", bbox_inches='tight')

    plot_best_comparison(data, model_names, save_path, color_dict)
    plot_model_sampling(data, model_names, save_path, color_dict, suffix_to_plot, best_suffix=best_suffix)

def plot_model_sampling(data, model_names, save_path, color_dict, suffix_to_plot, best_suffix=False):
    for (model, file) in data.keys():
        fig = plt.figure(figsize=(12, 6), dpi=300)
        method_data = data[(model, file)]
        search_method = file.split(".")[0].split("_")[0]
        if search_method != "bestofn":
            continue
        
        model_and_search_name = "_".join([model, search_method])

        method_data["suffix"] = method_data["metric"].apply(lambda x: x.split("-")[-1] if "-" in x else "")
        
        # Get unique base method names
        base_methods = sorted(set(m.split("-sum")[0].split("-min")[0].split("-last")[0].split("-orm_avg")[0] for m in method_data["metric"]))
        base_methods.remove("time")

        # Best method has the highest value
        possible_methods = ["pass_at_k", "majority_vote", "best_of_n-sum", "best_of_n-min", "best_of_n-last","best_of_n-orm_avg", "weighted_majority_vote-sum", "weighted_majority_vote-min", "weighted_majority_vote-last", "weighted_majority_vote-orm_avg"]

        metric_data = method_data[method_data["metric"].isin(possible_methods)]
        
        colors = {
            "majority_vote": "blue",
            "best_of_n": "orange",
            "weighted_majority_vote": "green",
            "pass_at_k": "red",
        }
        linestyles = {"sum": "solid", "min": "dashed", "last": "dotted", "orm_avg": "dashdot"}
        for current_method in metric_data["metric"].unique():
            sub_data = metric_data[metric_data["metric"] == current_method]
            tok_data = method_data[method_data["metric"] == "total_tokens"]

            if len(sub_data) != len(tok_data):
                sub_data = sub_data[:len(tok_data)]

            ax = fig.gca()
            sub_data = sub_data.sort_values("k")
            name = model_and_search_name
            if 'bestofn' in name:
                name = name.replace('bestofn', 'sampling')
            base_method = current_method.split("-")[0]
            suffix = current_method.split("-")[-1]
            # if suffix != "" and suffix not in suffix_to_plot:
            #     continue
            color = colors[base_method]
            if suffix in linestyles:
                linestyle = linestyles[suffix]
            else:
                linestyle = "solid"

            ax.errorbar(
                sub_data["k"],
                # tok_data["value"],
                sub_data["value"],
                yerr=sub_data["std"],
                label=f"{name} ({current_method})",
                linestyle=linestyle,
                color=color,
                marker="o",
                capsize=3
            )

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{model} sampling")
        ax.set_xlabel("k")
        ax.set_ylabel("Value")
        ax.set_xscale("log", base=2)
        # min_tok = min(tok_data["value"])
        # max_tok = max(tok_data["value"])
        # ax.set_xlim([min_tok//2, 2*max_tok])
        ax.grid(True)
        fig.savefig(f"{save_path}/{model}_sampling.png", bbox_inches='tight')


def plot_best_comparison(data, model_names, save_path, color_dict):
    fig = plt.figure(figsize=(12, 6), dpi=300)
    for (model, file) in data.keys():
        method_data = data[(model, file)]
        search_method = file.split(".")[0].split("_")[0]
        model_and_search_name = "_".join([model, search_method])

        method_data["suffix"] = method_data["metric"].apply(lambda x: x.split("-")[-1] if "-" in x else "")
        
        # Get unique base method names
        base_methods = sorted(set(m.split("-sum")[0].split("-min")[0].split("-last")[0].split("-orm_avg")[0] for m in method_data["metric"]))
        base_methods.remove("time")

        # Best method has the highest value
        possible_methods = ["majority_vote", "best_of_n-sum", "best_of_n-min", "best_of_n-last","best_of_n-orm_avg", "weighted_majority_vote-sum", "weighted_majority_vote-min", "weighted_majority_vote-last", "weighted_majority_vote-orm_avg"]

        metric_data = method_data[method_data["metric"].isin(possible_methods)]
        best_row_idx = metric_data.idxmax()["value"]
        best_row = metric_data.loc[best_row_idx]
        best_method = best_row["metric"]
        adv = best_row["adv"]
        if adv:
            model_and_search_name += "_adv"

        color = color_dict[model_and_search_name]

        sub_data = metric_data[metric_data["metric"] == best_method][metric_data["adv"] == adv]
        tok_data = method_data[method_data["metric"] == "total_tokens"][method_data["adv"] == adv]

        if len(sub_data) != len(tok_data):
            sub_data = sub_data[:len(tok_data)]

        ax = fig.gca()
        sub_data = sub_data.sort_values("k")
        name = model_and_search_name
        if 'bestofn' in name:
            name = name.replace('bestofn', 'sampling')
        ax.errorbar(
            sub_data["k"],
            # tok_data["value"],
            sub_data["value"],
            yerr=sub_data["std"],
            label=f"{name} ({best_method})",
            linestyle="solid",
            color=color,
            marker="o",
            capsize=3
        )

    ax.legend()  # Ensure legend is added if any labels exist
    ax.set_title("Best Comparison")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log", base=2)
    min_tok = min(tok_data["value"])
    max_tok = max(tok_data["value"])
    # ax.set_xlim([min_tok//2, 2*max_tok])
    ax.grid(True)
    fig.savefig(f"{save_path}/best_comparison.png", bbox_inches='tight')


if __name__ == "__main__":
    # model_names = ["qwen_genPPO_0.1" , "qwen_genPPO_0.3", "qwen_genPPO_0.5"]ad
    # model_names = ["qwen_ppo_math128", "qwen_genPPO_10_math128"]
    # model_names = ["0.3_aime", "0.8_aime"]
    # model_names = ["qwen_ORM_1_math128", "qwen_genPPO_10", "qwen_genPPO_0.8"]
    # model_names = ["qwen_ORM_1_aime", "qwen_ORM_2_aime", "qwen_ORM_4_aime", "qwen_ORM_8_aime", "qwen_ORM_16_aime"]
    # model_names = ["qwen_ORM_1_math128", "qwen_ORM_2_math128", "qwen_ORM_4_math128", "qwen_ORM_8_math128", "qwen_ORM_16_math128"]
    # model_names = ["simple_test", "clf_1_math128"]
    # model_names = ["clf_no_instr_init_math128", "sft_no_instr_1_math128"]
    model_names = ["grpo_scot_math_1_sft_2e-4__42_math128_best", "grpo_scot_math_1_clf_3e-4__42_math128", "grpo_scot_math_1_clf_1.5e-4__42_math128", "grpo_scot_math_0.3_sft_8e-5__42_math128"]
    # model_names = ["qwen7B_genPPO_0.3"]
    data = load_csvs(model_names)
    # plot_all(data, model_names, save_path="plots/overall", suffix_to_plot=["sum", "min", "last"], best_suffix = True)
    # plot_all(data, model_names, save_path="plots/qwen_ppo", suffix_to_plot=["sum", "min", "last", "orm_avg"])
    # plot_all(data, model_names, save_path="plots/compare_0.8_and_10", suffix_to_plot=["sum", "min", "last", "orm_avg"], best_suffix=True)
    # plot_all(data, model_names, save_path="plots/aime", suffix_to_plot=["sum", "min", "last", "orm_avg"], best_suffix=True)
    # plot_all(data, model_names, save_path="plots/orm_compare", suffix_to_plot=["sum", "min", "last", "orm_avg"], best_suffix=False)
    # plot_all(data, model_names, save_path="plots/orm_compare_aime", suffix_to_plot=["sum", "min", "last", "orm_avg"], best_suffix=False)

    # plot_all(data, model_names, save_path="plots/qwen7B", suffix_to_plot=["sum", "min", "last"], best_suffix=False)
    # plot_all(data, model_names, save_path="plots/ORM_aime", suffix_to_plot=["last"], best_suffix=False)
    # plot_all(data, model_names, save_path="plots/sft_clf_compare", suffix_to_plot=["last"], best_suffix=False)
    # plot_all(data, model_names, save_path="plots/grpo_aime", suffix_to_plot=["last"], best_suffix=False)
    plot_all(data, model_names, save_path="plots/grpo_idk", suffix_to_plot=["last"], best_suffix=False)