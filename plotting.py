import json
import matplotlib.pyplot as plt
import os
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
import numpy as np
from matplotlib import rcParams

rcParams.update({'font.size': 14})  # Increase base font size for the entire plot

# Step 1: Define the directory containing the JSON files and the file names
data_dir = "/home/mila/k/kusha.sareen/scratch/genPPO/outputs"  # Replace with the path where your JSON files are stored
json_files = ["20241129-232331_rebase", "20241129-230427_bestofn_vineppo", "20241129-081543_bestofn_ckpt2", "20241205-233059_beamsearch_ckpt2", "rebase_k_128"]
method_names = ["Rebase (GenPPO)", "Sampling (VinePPO)", "Sampling (GenPPO)", "Beam Search (GenPPO)", "Rebase (VinePPO + RM)"]  # Replace with actual method names
subcategory_names = ["prod", "min", "last"]  # Subcategories for Best-of-N and Weighted Majority Vote

# Step 2: Define consistent colors for methods and subcategories
base_colors = {
    "Rebase (GenPPO)": "#1f77b4",  # Blue
    "Sampling (VinePPO)": "#ff7f0e",  # Orange
    "Sampling (GenPPO)": "#2ca02c",  # Green
    "Beam Search (GenPPO)": "#d62728",  # Red
    "Rebase (VinePPO + RM)": "#9467bd",  # Purple
}
# Define a function to generate subcategory shades
def get_shades(base_color, num_shades):
    rgba = to_rgba(base_color)
    return [
        (rgba[0] + 1 * (1 - rgba[0]) * (i / (num_shades + 1)),
         rgba[1] + 1 * (1 - rgba[1]) * (i / (num_shades + 1)),
         rgba[2] + 1 * (1 - rgba[2]) * (i / (num_shades + 1)),
         1.0)
        for i in range(1, num_shades + 1)
    ]


# Generate shades for subcategories
subcategory_shades = {
    method: get_shades(base_colors[method], len(subcategory_names))
    for method in method_names
}

# Step 3: Load the JSON data into a dictionary
data = {}
for file in json_files:
    with open(os.path.join(data_dir, file), "r") as f:
        method_name = file.split(".")[0]  # Use the file name (without extension) as the method label
        data[method_name] = json.load(f)

# Step 4: Define a helper function to plot line graphs for a specific category
def get_first_n_powers_of_two(n):
    return [2 ** i for i in range(n)]

# small patch
total_tokens = data['20241129-081543_bestofn_ckpt2']['total_tokens']
ks = get_first_n_powers_of_two(7)
data['20241129-081543_bestofn_ckpt2']['total_tokens'] = {k: total_tokens / 200 * k for k in ks}

def plot_category(data, category, subcategories=False, title=None, xlabel="N", ylabel="Accuracy", yscale="linear", figsize=(10, 6)):
    """
    Plot a category of metrics with optional subcategories (e.g., sum, min, last).
    """
    print(f"Plotting {category}...")
    plt.figure(figsize=figsize)
    for method, content in data.items():
        method_idx = list(data.keys()).index(method)
        method_name = method_names[method_idx]

        if subcategories:
            # Plot subcategories (e.g., sum, min, last)
            subcats = list(content[category].keys())
            for subcat in subcats:
                y_values = list(content[category][subcat].values())
                x_values = get_first_n_powers_of_two(len(y_values))
                subcat_idx = subcats.index(subcat)
                subcat_name = subcategory_names[subcat_idx]
                # Use subcategory shades for consistent coloring
                color = subcategory_shades[method_name][subcat_idx]
                if method_name == "Sampling (VinePPO)":
                    subcat_name = "solution"

                plt.plot(x_values, y_values, label=f"{method_name} ({subcat_name})", color=color)
        else:
            # Plot main category (e.g., pass_at_k)
            x_values = get_first_n_powers_of_two(len(content[category]))
            y_values = list(content[category].values())
            plt.plot(x_values, y_values, label=method_name, color=base_colors[method_name])

    # Formatting the plot
    plt.title(title if title else category.replace("_", " ").capitalize(), fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xscale("log", base=2)  # Set x-axis to logarithmic scale for better clarity
    plt.yscale(yscale)
    if subcategories:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(fontsize=12)
    # plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/{category}.png", dpi=300)  # Save the plot as an image file

def plot_best_comparison(data, title="Scaling Test-Time Compute", ylabel='Accuracy', xlabel='N', yscale='linear'):
    print(f"Plotting best comparison...")
    plt.figure(figsize=(12, 6))
    for method, content in data.items():
        method_idx = list(data.keys()).index(method)
        method_name = method_names[method_idx]
        category = best_category_dict[method_name]
        color = base_colors[method_name]

        if has_subcategories_dict[category]:
            # Plot subcategories (e.g., sum, min, last)
            subcats = list(content[category].keys())
            subcat = best_subcategory_dict[method_name]
            y_values = list(content[category][subcat].values())
            x_values = get_first_n_powers_of_two(len(y_values))
            subcat_idx = subcats.index(subcat)
            subcat_name = subcategory_names[subcat_idx]
            # Use subcategory shades for consistent coloring
            if method_name == "Sampling (VinePPO)":
                subcat_name = "solution"


            plt.plot(x_values, y_values, label=f"{category_to_name[category]} ({subcat_name}) + {method_name}", color=color)
        else:
            # Plot main category (e.g., pass_at_k)
            x_values = get_first_n_powers_of_two(len(content[category]))
            y_values = list(content[category].values())
            plt.plot(x_values, y_values, label=f"{category_to_name[category]} + {method_name}", color=color)

    # Formatting the plot
    plt.title(title if title else category.replace("_", " ").capitalize(), fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xscale("log", base=2)  # Set x-axis to logarithmic scale for better clarity
    plt.yscale(yscale)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/best_comparison.png", dpi=300)  # Save the plot as an image file

# Step 5: Plot each category
# Pass@k
plot_category(data, "pass_at_k", title="Pass@N", ylabel='Pass@N')

# Best-of-N with subcategories (sum, min, last)
plot_category(data, "best_of_n", subcategories=True, title="Best-of-N")

# Majority Vote
plot_category(data, "majority_vote", title="Majority Vote", figsize=(6, 6))

# Weighted Majority Vote with subcategories (sum, min, last)
plot_category(
    data,
    "weighted_majority_vote",
    subcategories=True,
    title="Weighted Majority Vote",
)

plot_category(data, "total_tokens", title="Total Tokens", ylabel="Total Tokens", yscale="log")

has_subcategories_dict = {'best_of_n': True, 'weighted_majority_vote': True, 'majority_vote': False}
best_subcategory_dict = {'Rebase (GenPPO)': 'min', 'Sampling (VinePPO)': 'sum', 'Sampling (GenPPO)': 'last', 'Beam Search (GenPPO)': 'min', 'Rebase (VinePPO + RM)': 'min'}
best_category_dict = {'Rebase (GenPPO)': 'weighted_majority_vote', 'Sampling (VinePPO)': 'majority_vote', 'Sampling (GenPPO)': 'weighted_majority_vote', 'Beam Search (GenPPO)': 'best_of_n', 'Rebase (VinePPO + RM)': 'weighted_majority_vote'}
category_to_name = {'pass_at_k': 'Pass@N', 'best_of_n': 'Best-of-N', 'majority_vote': 'Majority Vote', 'weighted_majority_vote': 'Weighted Majority Vote', 'total_tokens': 'Total Tokens'}
# check sampling vineppo

plot_best_comparison(data, title="Best Test-Time Compute", ylabel='Accuracy', xlabel='N', yscale='linear')