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
    assert len(method_names) <= 9, "Only 9 base colors are supported"

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
    ]

    return {method: colors[i] for i, method in enumerate(method_names)}

def plot_method(method_data, model, file, fig_maj, fig_bon, fig_wmaj, fig_pan, fig_tok, color_dict, suffix_to_plot, only_best_suffix=False):
    # Extract the metric names
    search_method = file.split(".")[0].split("_")[0]
    method_data["suffix"] = method_data["metric"].apply(lambda x: x.split("_")[-1] if "_" in x else "")
    
    # Define linestyle mapping
    linestyle_map = {"sum": "solid", "min": "dashed", "last": "dotted"}
    
    # Define figure mapping for different plot types
    fig_map = {
        "majority_vote": fig_maj,
        "best_of_n": fig_bon,
        "weighted_majority_vote": fig_wmaj,
        "pass_at_k": fig_pan,
        "total_tokens": fig_tok,
    }

    # Get unique base method names
    base_methods = sorted(set(m.split("_sum")[0].split("_min")[0].split("_last")[0] for m in method_data["metric"]))
    base_methods.remove("time")

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
            
            # Determine which figure to use
            key = base_method
            fig = fig_map.get(key)

            ax = fig.gca()

            if sub_data["suffix"].nunique() == 1:
                # sort by true_k
                sub_data = sub_data.sort_values("true_k")
                ax.errorbar(
                    sub_data["true_k"],
                    sub_data["value"],
                    yerr=sub_data["std"],
                    label=f"{model_and_search_name}",
                    linestyle="solid",
                    color=color,
                    marker="o",
                    capsize=3
                )
                print(key, model_and_search_name)
                ax.legend()  # Ensure legend is added if any labels exist
                ax.set_title(key)
                ax.set_xlabel("k")
                ax.set_ylabel("Value")
                ax.set_xscale("log", base=2)
                if key == "total_tokens":
                    ax.set_yscale("log", base=2)
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
                    # sort by true_k
                    sub_sub_data = sub_sub_data.sort_values("true_k")
                    ax.errorbar(
                        sub_sub_data["true_k"],
                        sub_sub_data["value"],
                        yerr=sub_sub_data["std"],
                        label=f"{model_and_search_name} ({suffix})",
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
                    ax.grid(True)

    return fig_maj, fig_bon, fig_wmaj, fig_pan, fig_tok

def flatten_list(l):
    return [item for sublist in l for item in sublist]
    
def plot_all(data, model_names, save_path="plots", suffix_to_plot=["sum", "min", "last"], best_suffix=False):
    model_and_search_names  = [[f"{model}_rebase", f"{model}_rebase_adv", f"{model}_bestofn"] for model in model_names]
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

def plot_best_comparison(data, model_names, save_path, color_dict):
    fig = plt.figure(figsize=(12, 6), dpi=300)
    for (model, file) in data.keys():
        method_data = data[(model, file)]
        search_method = file.split(".")[0].split("_")[0]
        model_and_search_name = "_".join([model, search_method])

        method_data["suffix"] = method_data["metric"].apply(lambda x: x.split("_")[-1] if "_" in x else "")
        
        # Get unique base method names
        base_methods = sorted(set(m.split("_sum")[0].split("_min")[0].split("_last")[0] for m in method_data["metric"]))
        base_methods.remove("time")

        # Best method has the highest value
        possible_methods = ["majority_vote", "best_of_n_sum", "best_of_n_min", "best_of_n_last","weighted_majority_vote_sum", "weighted_majority_vote_min", "weighted_majority_vote_last"]

        metric_data = method_data[method_data["metric"].isin(possible_methods)]
        best_row_idx = metric_data.idxmax()["value"]
        best_row = metric_data.loc[best_row_idx]
        best_method = best_row["metric"]
        adv = best_row["adv"]
        if adv:
            model_and_search_name += "_adv"

        color = color_dict[model_and_search_name]

        sub_data = metric_data[metric_data["metric"] == best_method][metric_data["adv"] == adv]
        ax = fig.gca()
        sub_data = sub_data.sort_values("true_k")
        ax.errorbar(
            sub_data["true_k"],
            sub_data["value"],
            yerr=sub_data["std"],
            label=f"{model_and_search_name} ({best_method})",
            linestyle="solid",
            color=color,
            marker="o",
            capsize=3
        )

    ax.legend()  # Ensure legend is added if any labels exist
    ax.set_title("Best Comparison")
    ax.set_xlabel("k")
    ax.set_ylabel("Value")
    ax.set_xscale("log", base=2)
    ax.grid(True)
    fig.savefig(f"{save_path}/best_comparison.png", bbox_inches='tight')


if __name__ == "__main__":
    model_names = ["qwen_genPPO_0.1" , "qwen_genPPO_0.3", "qwen_genPPO_0.5"]
    # model_names = ["qwen_genPPO_0.1"]
    data = load_csvs(model_names)
    plot_all(data, model_names, save_path="plots/overall", suffix_to_plot=["sum", "min", "last"], best_suffix = True)
    # plot_all(data, model_names, save_path="plots/qwen_genPPO_0.1", suffix_to_plot=["sum", "min", "last"])
