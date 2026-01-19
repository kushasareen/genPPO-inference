import argparse
import numpy as np
import os
import json
import pandas as pd

def get_data(model):
    path = f"/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{model}"
    json_files = os.listdir(path)
    data = {}
    for file in json_files:
        # Check if the file is a JSON file (and not a directory)
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and not file.lower().endswith('.csv'):
            with open(file_path, "r") as f:
                data[file] = json.load(f)
    return data

def parse_data(data):
    metrics = {}
    true_k = {}
    
    for key, content in data.items():
        split = key.split('_')
        method = content['config']['search_algorithm']
        
        k = content['config']['top_k']
        seed = content['config']['seed']
        adv = content['config']['use_advantage']

        if method != "rebase":
            bestofn_to_df(content, method, k, adv)
            continue
        
        for metric, values in content.items():
            if metric == "config" or metric == "base_results":
                continue
            
            first_key = list(values.keys())[0]
            if not isinstance(values[first_key], dict):
                keys = list(values.keys()) # remove all text
                int_keys = [int(key.split('@')[-1]) for key in keys]
                max_key = keys[int_keys.index(max(int_keys))]
                max_key_value = values[max_key]
                max_int_key = int(max_key.split('@')[-1])
                if max_int_key == 1 and metric != "total_tokens":
                    max_key_value = content["base_results"]["top 1"]


                key_tuple = (metric, k, adv)
                if key_tuple not in metrics:
                    metrics[key_tuple] = []
                    true_k[key_tuple] = []
                metrics[key_tuple].append(max_key_value)
                true_k[key_tuple].append(max_int_key)

            else:  # Metrics with nested sub-metrics
                for sub_metric, sub_values in values.items():
                    keys = list(sub_values.keys())
                    int_keys = [int(key.split('@')[-1]) for key in keys]
                    max_key = keys[int_keys.index(max(int_keys))]
                    max_key_value = sub_values[max_key]
                    max_int_key = int(max_key.split('@')[-1])
                    if max_int_key == 1 and metric != "total_tokens":
                        max_key_value = content["base_results"]["top 1"]
                        
                    metric_name = f'{metric}-{sub_metric}'
                    key_tuple = (metric_name, k, adv)
                    if key_tuple not in metrics:
                        metrics[key_tuple] = []
                        true_k[key_tuple] = []
                    metrics[key_tuple].append(max_key_value)
                    true_k[key_tuple].append(max_int_key)
                
    # Compute averages
    aggregated_results = {}
    aggregated_std = {}
    aggregated_true_k = {}
    aggregated_true_k_std = {}
    for key_tuple, values in metrics.items():
        aggregated_results[key_tuple] = np.mean(values)
        aggregated_std[key_tuple] = np.std(values)
        aggregated_true_k[key_tuple] = np.mean(true_k[key_tuple])
        aggregated_true_k_std[key_tuple] = np.std(true_k[key_tuple])
    
    # Convert to DataFrame  
    df = convert_to_df(aggregated_results, aggregated_true_k, aggregated_std, aggregated_true_k_std)
    df.to_csv(f"/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{args.model}/rebase_results.csv", index=False)

def bestofn_to_df(content, method, k, adv):
    assert method == "bestofn"
    # save content to csv in same format as rebase
    
    rows = []
    for metric, values in content.items():
        if metric == "config" or metric == "base_results":
            continue
        
        first_key = list(values.keys())[0]
        if not isinstance(values[first_key], dict):  # Metrics with multiple k-values (like pass_at_k)
            for key, value in values.items():
                k_int = int(key.split('@')[-1])
                rows.append({
                    'metric': metric,
                    'k': k_int,
                    'adv': False,
                    'value': value,
                    'true_k': k_int,
                    'std': 0,
                    'true_k_std': 0
                })
        else:  # Metrics with nested sub-metrics
            for sub_metric, sub_values in values.items():
                metric_name = f'{metric}-{sub_metric}'
                for key, value in sub_values.items():
                    k_int = int(key.split('@')[-1])
                    rows.append({
                        'metric': metric_name,
                        'k': k_int,
                        'adv': False,
                        'value': value,
                        'true_k': k_int,
                        'std': 0,
                        'true_k_std': 0
                    })

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    # Optional: Reorder columns for clarity
    df = df[['metric', 'k', 'adv', 'value', 'true_k', 'std', 'true_k_std']]
    df.to_csv(f"/home/mila/k/kusha.sareen/scratch/genPPO/outputs/{args.model}/bestofn_results.csv", index=False)

def convert_to_df(data, true_k, std, true_k_std):
    rows = []
    for (metric, k, adv), value in data.items():
        rows.append({
            'metric': metric,
            'k': k,
            'adv': adv,
            'value': value,
            'true_k': true_k[(metric, k, adv)],
            'std': std[(metric, k, adv)],
            'true_k_std': true_k_std[(metric, k, adv)]
        })

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    # Optional: Reorder columns for clarity
    df = df[['metric', 'k', 'adv', 'value', 'true_k', 'std', 'true_k_std']]
    return df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen_genPPO_0.5')
    args = parser.parse_args()

    data = get_data(args.model)
    parse_data(data)
