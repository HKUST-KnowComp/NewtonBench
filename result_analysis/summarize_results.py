import argparse
import os
import json
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from typing import List

def extract_version_from_path(results_dir):
    """Extract version (v10, v15, etc.) from results directory path"""
    match = re.search(r'v(\d+)$', results_dir.rstrip('/'))
    if match:
        return f"v{match.group(1)}"
    else:
        return "v_unknown"  # fallback
    

def read_models_from_file(models_file: Path) -> List[str]:
    if not models_file.exists():
        return []
    models: List[str] = []
    with models_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    return models


def update_results(model_name, result_dir):
    """Update the results_by_trial.csv file with the latest trial results."""
    csv_path = 'result_analysis/results_by_trial.csv'

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            'trial_id', 'module', 'model_name', 'noise_level', 'equation_difficulty', 'model_system',
            'law_version', 'agent_backend', 'rmsle', 'exact_accuracy', 'rounds',
            'experiments', 'total_tokens', 'file_version'
        ])

    model_dir = os.path.join(result_dir, model_name)
    if not os.path.isdir(model_dir):
        print(f"Directory not found for model: {model_name}")
        return

    modules_to_process = os.listdir(model_dir)

    for module in modules_to_process:
        module_path = os.path.join(model_dir, module)
        if not os.path.isdir(module_path):
            continue

        for root, dirs, files in os.walk(module_path):
            if 'trials' in dirs:
                trials_dir = os.path.join(root, 'trials')
                for file in os.listdir(trials_dir):
                    if file.endswith('.json') and 'fail' not in file:
                        trial_id_match = re.search(r'trial(\d+)', file)
                        if not trial_id_match:
                            continue
                        trial_id = int(trial_id_match.group(1))

                        file_path = os.path.join(trials_dir, file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        file_version = extract_version_from_path(root)

                        new_row = {
                            'trial_id': trial_id,
                            'module': data.get('module_name'),
                            'model_name': data.get('model_name'),
                            'noise_level': data.get('noise_level'),
                            'equation_difficulty': data.get('equation_difficulty'),
                            'model_system': data.get('model_system'),
                            'law_version': data.get('law_version'),
                            'agent_backend': data.get('agent_backend'),
                            'rmsle': data.get('evaluation', {}).get('rmsle'),
                            'exact_accuracy': data.get('evaluation', {}).get('exact_accuracy'),
                            'rounds': data.get('rounds'),
                            'experiments': data.get('num_experiments'),
                            'total_tokens': data.get('total_tokens'),
                            'file_version': file_version
                        }

                        # Check if the trial already exists and update it
                        mask = (df['trial_id'] == new_row['trial_id']) & \
                                (df['module'] == new_row['module']) & \
                                (df['model_name'] == new_row['model_name']) & \
                                (df['noise_level'] == new_row['noise_level']) & \
                                (df['equation_difficulty'] == new_row['equation_difficulty']) & \
                                (df['model_system'] == new_row['model_system']) & \
                                (df['law_version'] == new_row['law_version']) & \
                                (df['agent_backend'] == new_row['agent_backend'])

                        if mask.any():
                            df.loc[mask, new_row.keys()] = new_row.values()
                        else:
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Results updated in {csv_path}")


def detect_outliers_modified_zscore_column(df, column_name, threshold=3.5):
    """
    Detect outliers in a DataFrame column using Modified Z-Score method and mask them as NaN.
    Args:
        df: pandas DataFrame
        column_name: name of the column to process
        threshold: Modified Z-Score threshold (default 3.5)
    Returns:
        df: DataFrame with outliers in the specified column masked as NaN
    """
    data = df[column_name].values

    if len(data) == 0:
        return df

    # Replace inf with nan for median calculation
    data_for_stats = np.where(np.isinf(data), np.nan, data)

    median = np.nanmedian(data_for_stats)
    mad = np.nanmedian(np.abs(data_for_stats - median))

    if mad == 0:
        outlier_mask = ~np.isfinite(data)
    else:
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_mask |= ~np.isfinite(modified_z_scores)

    # Mask outliers as NaN in the original DataFrame
    df.loc[outlier_mask, column_name] = np.nan

    return df

def calculate_trial_stats(df):
    """
    Calculates statistics based on trial performance.
    1. Groups by trial_id and calculates the mean accuracy/rmsle for each trial.
    2. Calculates the mean and std of those per-trial means.
    """
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan
    
    # Step 1: Group by trial and calculate mean for each trial
    trial_means = df.groupby('trial_id').agg(
        mean_accuracy=('exact_accuracy', 'mean'),
        mean_rmsle=('rmsle', 'mean')
    ).dropna()

    if trial_means.empty:
        return np.nan, np.nan, np.nan, np.nan

    # Step 2: Calculate mean and std of the per-trial means
    final_mean_acc = trial_means['mean_accuracy'].mean()
    final_std_acc = trial_means['mean_accuracy'].std()
    final_mean_rmsle = trial_means['mean_rmsle'].mean()
    final_std_rmsle = trial_means['mean_rmsle'].std()
    
    return final_mean_acc, final_std_acc if not np.isnan(final_std_acc) else 0, final_mean_rmsle, final_std_rmsle if not np.isnan(final_std_rmsle) else 0


def aggregate_results(output_csv, model_name=None):
    """
    Main function to generate the aggregated results summary.
    """
    print("Loading all trial results into DataFrame...")
    csv_path = 'result_analysis/results_by_trial.csv'
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No valid results found to process.")
        return
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"Loaded {len(df)} records.")

    if model_name:
        df = df[df['model_name'] == model_name]
        print(f"Filtered for model: {model_name}")

    print("Applying outlier detection per module, equation_difficulty, model_system combination and agent backend for RMSLE...")
    
    # Get unique groups
    groups = df.groupby(['module', 'equation_difficulty', 'model_system', 'agent_backend'])
    
    # Create a new dataframe with outliers removed
    cleaned_df_list = []
    count = 0
    for name, group in groups:
        cleaned_group = detect_outliers_modified_zscore_column(group.copy(), 'rmsle')
        cleaned_df_list.append(cleaned_group)
        count+=1
    
    if cleaned_df_list:
        df = pd.concat(cleaned_df_list).reset_index(drop=True)

    difficulties = ["easy", "medium", "hard"]
    systems = ["vanilla_equation", "simple_system", "complex_system"]

    aggregated_data = []
    
    backends = sorted(df['agent_backend'].unique(), key=lambda x: (x != 'llm_explore', x))
    models = sorted(df['model_name'].unique())

    for model in models:
        for backend in backends:
            row = {'model_name': model, 'agent_backend': backend}
            
            model_backend_df = df[(df['model_name'] == model) & (df['agent_backend'] == backend)]
            if model_backend_df.empty:
                continue

            for system in systems:
                for difficulty in difficulties:
                    col_name = f"acc_{difficulty}_{system}"
                    
                    filtered_df = model_backend_df[
                        (model_backend_df['equation_difficulty'] == difficulty) & 
                        (model_backend_df['model_system'] == system)
                    ]
                    
                    mean_acc, std_acc, _, _ = calculate_trial_stats(filtered_df)
                    
                    if pd.notna(mean_acc):
                        row[col_name] = f"{(mean_acc*100):.1f} (±{(std_acc*100):.3f})"
                    else:
                        row[col_name] = "N/A"

            overall_mean_acc, overall_std_acc, overall_mean_rmsle, overall_std_rmsle = calculate_trial_stats(model_backend_df)
            
            if pd.notna(overall_mean_acc):
                row['overall_acc'] = f"{(overall_mean_acc*100):.1f} (±{(overall_std_acc*100):.3f})"
            else:
                row['overall_acc'] = "N/A"
            
            if pd.notna(overall_mean_rmsle):
                row['overall_rmsle'] = f"{overall_mean_rmsle:.4f} (±{overall_std_rmsle:.4f})"
            else:
                row['overall_rmsle'] = "N/A"

            avg_total_tokens = model_backend_df['total_tokens'].mean()

            if pd.notna(avg_total_tokens):
                row['avg_total_tokens'] = f"{avg_total_tokens:.0f}"
            else:
                row['avg_total_tokens'] = "N/A"
            
            aggregated_data.append(row)

    summary_df = pd.DataFrame(aggregated_data)

    column_order = ['model_name', 'agent_backend'] + [
        f"acc_{diff}_{system}" 
        for system in systems
        for diff in difficulties
    ] + ['overall_acc', 'overall_rmsle', 'avg_total_tokens']
    
    summary_df = summary_df.reindex(columns=column_order)

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv, encoding='utf-8')
        # Use a temporary key for merging to avoid issues with multi-index
        summary_df['__key'] = summary_df['model_name'] + '_' + summary_df['agent_backend']
        existing_df['__key'] = existing_df['model_name'] + '_' + existing_df['agent_backend']
        
        # Filter out rows from existing_df that will be updated
        rows_to_keep = existing_df[~existing_df['__key'].isin(summary_df['__key'])]
        
        # Combine the rows to keep with the new data
        final_df = pd.concat([rows_to_keep, summary_df], ignore_index=True)
        
        # Clean up the temporary key
        final_df.drop(columns=['__key'], inplace=True)
        print(f"Successfully updated aggregated summary at: {output_csv}")
    else:
        final_df = summary_df
        print(f"Successfully generated aggregated summary at: {output_csv}")

    final_df.to_csv(output_csv, index=False, encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and aggregate all trial results into a summary CSV.")
    parser.add_argument("-m", "--model_name", default="all", help="LLMs to be processed")
    parser.add_argument("-d", "--result_dir", default="evaluation_results", help="Directory containing the evaluation results.")
    parser.add_argument("-o", "--output_csv", default="result_analysis/aggregated_trial_summary.csv", help="Path to save the final aggregated CSV file.")
    parser.add_argument("--models_file", type=str, default="configs/models.txt", help="Path to newline-delimited models list when --model_name is not given.")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name == "all":
        all_models = read_models_from_file(Path(args.models_file))
        for model in all_models:
            print(f"--- Processing model: {model} ---")
            update_results(model, args.result_dir)
            aggregate_results(args.output_csv, model)
        print("--- Finished processing all models. ---")
    else:
        print(f"--- Processing model: {args.model_name} ---")
        update_results(args.model_name, args.result_dir)
        aggregate_results(args.output_csv, args.model_name)
        print(f"--- Finished processing model: {args.model_name} ---")
