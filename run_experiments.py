import os
import json
import argparse
import time
import sys
import importlib
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import traceback
import pandas as pd

from utils.vanilla_agent import conduct_exploration
import warnings
warnings.filterwarnings("ignore")

try:
	from utils.code_assisted_agent import conduct_code_assisted_exploration
	_WITH_CODE_ASSISTANCE = True
except Exception:
	_WITH_CODE_ASSISTANCE = False

def format_chat_history(chat_history):
    """Format chat history as a readable log file."""
    lines = []
    for i, msg in enumerate(chat_history):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        lines.append(f"--- Round {i+1} ({role}) ---\n{content}\n")
    return '\n'.join(lines)

def write_fail_result_with_retries(args, final_error, retry_history, func_sig):
    trial_id, noise_level, model_name, module_name, difficulty, system, law_version, trial_dir, max_retries, judge_model_name, agent_backend = args
    fail_result = {
        "trial_id": trial_id,
        "module_name": module_name,
        "noise_level": noise_level,
        "model_name": model_name,
        "equation_difficulty": difficulty,
        "model_system": system,
        "law_version": law_version,
        "retry_attempts": max_retries,
        "LLM judge": judge_model_name,
        "agent_backend": agent_backend,
        "retry_history": retry_history,
        "error": final_error,
        "status": "failed",
        "submitted_law": f"{func_sig} return float('nan')",
        "rounds": 0,
        "total_tokens": 0,
        "num_experiments": 0,
        "chat_history": [],
        "evaluation": {
            "rmsle": float("nan"),
            "exact_accuracy": 0.0,
            "symbolic_equivalent": False,
            "symbolic_msg": f"Trial failed: {final_error}",
            "error": final_error
        }
    }
    trial_json_path = os.path.join(trial_dir, f"trial{trial_id}_fail.json")
    try:
        with open(trial_json_path, 'w', encoding='utf-8') as f:
            json.dump(fail_result, f, indent=2)
    except Exception as e:
        print(f"[Trial {trial_id} ERROR] Could not write fail trial json: {e}")
    return fail_result

def extract_version_from_path(results_dir):
    """Extract version (v10, v15, etc.) from results directory path"""
    # Extract from path like: evaluation_results/dsr1/m0_gravity/vanilla_agent/easy/v0/function_noise0_0_v2/
    match = re.search(r'v(\d+)$', results_dir.rstrip('/'))
    if match:
        return f"v{match.group(1)}"
    else:
        return "v0"  # fallback

def run_trial(args):
    """
    A single worker function for one trial with retry logic. It is module-agnostic.
    It expects every module to have the following interface:
    - A `TASK_PROMPT` string.
    - A `run_experiment_for_module(...)` function.
    - An `evaluate_law(str)` function.
    """
    trial_id, noise_level, model_name, module_name, difficulty, system, law_version, trial_dir, max_retries, judge_model_name, agent_backend = args
    print(f"Starting trial {trial_id} for module '{module_name}' with {model_name}, noise {noise_level} (equation difficulty: {difficulty}, model system: {system}, law version: {law_version}, backend: {agent_backend}")
    
    retry_history = []
    final_error = None
    
    for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
        try:
            if attempt > 0:
                print(f"[Trial {trial_id} RETRY] Attempt {attempt + 1}/{max_retries + 1}")
                # Add small delay between retries to avoid overwhelming APIs
                time.sleep(2)
            
            # Dynamically import the specified module for this process
            module = importlib.import_module(f"modules.{module_name}")
            
            # 1. Let the LLM explore and discover the law for the given module
            trial_info = {
                'trial_id': trial_id,
                'trial_dir': trial_dir
            }
            if agent_backend == "code_assisted_agent" and _WITH_CODE_ASSISTANCE:
                exploration_result = conduct_code_assisted_exploration(
                    module=module,
                    model_name=model_name,
                    noise_level=noise_level,
                    difficulty=difficulty,
                    system=system,
                    law_version=law_version,
                    trial_info=trial_info
                )
            else:
                exploration_result = conduct_exploration(
                    module=module,
                    model_name=model_name,
                    noise_level=noise_level,
                    difficulty=difficulty,
                    system=system,
                    law_version=law_version,
                    trial_info=trial_info
                )

            # 2. Evaluate the submitted law using the module's specific evaluator
            evaluation_metrics = module.evaluate_law(
                exploration_result["submitted_law"], 
                param_description=module.PARAM_DESCRIPTION, 
                difficulty=difficulty, 
                law_version=law_version, 
                judge_model_name=judge_model_name, 
                trial_info=trial_info
            )

            # 3. Combine all results for this trial
            final_result = {
                "trial_id": trial_id,
                "module_name": module_name,
                "noise_level": noise_level,
                "model_name": model_name,
                "equation_difficulty": difficulty,
                "model_system": system,
                "law_version": law_version,
                "retry_attempts": attempt,
                "LLM judge": judge_model_name,
                "agent_backend": agent_backend,
                "retry_history": retry_history,
                **exploration_result,
                "evaluation": evaluation_metrics
            }

            acc = evaluation_metrics.get('exact_accuracy', 0)
            if attempt > 0:
                print(f"[Trial {trial_id} SUCCESS] Retry attempt {attempt + 1} succeeded. Exact Accuracy: {'Yes' if acc == 1.0 else 'No'}")
            else:
                print(f"Finished trial {trial_id}. Exact Accuracy: {'Yes' if acc == 1.0 else 'No'}")
            
            # Write trial result and chat history to files
            trial_json_path = os.path.join(trial_dir, f"trial{trial_id}.json")
            with open(trial_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2)
            chat_log_path = os.path.join(trial_dir, f"trial{trial_id}_chat_history.log")
            with open(chat_log_path, 'w', encoding='utf-8') as f:
                f.write(format_chat_history(exploration_result.get("chat_history", [])))

            # Return full result for aggregation
            return final_result
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            tb_str = traceback.format_exc()
            
            # Record this attempt in retry history
            retry_record = {
                "attempt": attempt + 1,
                "error_type": error_type,
                "error_message": error_msg,
                "traceback": tb_str,
                "timestamp": datetime.now().isoformat()
            }
            retry_history.append(retry_record)
            
            print(f"[Trial {trial_id} ATTEMPT {attempt + 1}] {error_msg}")
            traceback.print_exc()
            
            # If this was the last attempt, return fail result
            if attempt == max_retries:
                print(f"[Trial {trial_id} FAILED] All {max_retries + 1} attempts exhausted")
                final_error = error_msg
                break
            else:
                print(f"[Trial {trial_id} RETRYING] Attempt {attempt + 1} failed, will retry...")
    
    # If we get here, all retries were exhausted
    return write_fail_result_with_retries(args, final_error, retry_history, module.FUNCTION_SIGNATURE)

def run_experiment_for_version(cli_args, module, law_version, num_trials):
    """Runs a full experiment for a single law version."""
    
    print(f"--- Running experiment for law_version: {law_version} with {num_trials} trials ---")

    # Create unique results directory with hierarchical structure
    noise_str = str(cli_args.noise).replace('.', '_')
    law_version_str = law_version if law_version is not None else "random"
       
    base_dir = os.path.join("evaluation_results", cli_args.model_name, cli_args.module, cli_args.agent_backend, cli_args.equation_difficulty, law_version_str)
    
    version_num = 1
    while True:
        experiment_name = f"{cli_args.model_system}_noise{noise_str}_v{version_num}"
        full_path = os.path.join(base_dir, experiment_name)
        if not os.path.exists(full_path):
            break
        version_num += 1
    results_dir = os.path.join(base_dir, experiment_name)
    trials_dir = os.path.join(results_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    start_time = time.time()
    
    max_retries = 3
    judge_model_name = "gpt41"
    
    pool_args = [
        (i, cli_args.noise, cli_args.model_name, cli_args.module, cli_args.equation_difficulty, cli_args.model_system, law_version, trials_dir, max_retries, judge_model_name, cli_args.agent_backend)
        for i in range(num_trials)
    ]
    
    # Run trials with dynamic batch processing based on CPU count
    actual_cpu_count = cpu_count()
    batch_size = min(actual_cpu_count, cli_args.trials)

    # Run trials with fixed batch processing 
    # batch_size = 6  # Fixed batch size
    num_batches = (num_trials + batch_size - 1) // batch_size
    
    # print(f"CPU Count: {actual_cpu_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Processes: {batch_size}")
    print(f"Number of Batches: {num_batches}")
    print(f"Total Trials: {num_trials}")
    
    results = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_trials)
        batch_args = pool_args[start_idx:end_idx]
        
        actual_batch_size = len(batch_args)
        
        print(f"Processing batch {batch_num + 1}/{num_batches} (trials {start_idx + 1}-{end_idx}) with {actual_batch_size} processes")
        
        with Pool(processes=actual_batch_size) as pool:
            batch_results = pool.map(run_trial, batch_args)
            results.extend(batch_results)
        
        print(f"Completed batch {batch_num + 1}/{num_batches}")

    end_time = time.time()

    # --- Aggregate and Print Final Results ---
    # Filter out any trials that failed due to errors
    valid_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    all_results = valid_results + failed_results
    
    if not valid_results:
        print(f"\nAll trials for law_version '{law_version}' failed. Please check the logs in '{results_dir}'.")
        return

    # Calculate metrics for all trials (including failed ones)
    all_rmsle_scores = np.array([r['evaluation']['rmsle'] for r in all_results])
    all_accuracies = np.array([r['evaluation']['exact_accuracy'] for r in all_results])
    all_turns = np.array([r['rounds'] for r in all_results if "error" not in r])
    all_experiments_used = np.array([r['num_experiments'] for r in all_results if "error" not in r])
    all_tokens_used = np.array([r['total_tokens'] for r in all_results if "error" not in r])

    # Filter out non-finite values for accurate statistics
    finite_all_rmsle = all_rmsle_scores[np.isfinite(all_rmsle_scores)]
    
    # Calculate retry statistics for all trials
    all_retry_attempts = [r.get('retry_attempts', 0) for r in all_results]
    total_retries = sum(all_retry_attempts)
    trials_with_retries = sum(1 for attempts in all_retry_attempts if attempts > 0)

    print("\n" + "="*50)
    print("BENCHMARK COMPLETED")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Module Tested: {cli_args.module}")
    print(f"  - Model Name: {cli_args.model_name}")
    print(f"  - Noise Level: {cli_args.noise}")
    print(f"  - Equation Difficulty: {cli_args.equation_difficulty}")
    print(f"  - Law Version: {law_version}")
    print(f"  - Model System: {cli_args.model_system}")
    print(f"  - Max Retries per Trial: {max_retries}")
    print(f"  - LLM Judge: {judge_model_name}")
    print(f"  - Total Trials: {num_trials} ({len(valid_results)} successful)")
    print(f"  - Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"  - Backend: {cli_args.agent_backend}")
    print("-"*50)
    print("Aggregated Results (All Trials):")
    print(f"  - Average Raw RMSLE: {np.nanmean(finite_all_rmsle):.4f}")
    print(f"  - Average Exact Accuracy: {np.mean(all_accuracies):.2%}")
    print(f"  - Average Rounds to Completion: {np.mean(all_turns):.2f}")
    print(f"  - Average Experiments Used per Trial: {np.mean(all_experiments_used):.2f}")
    print(f"  - Average Total Tokens Used per Trial: {np.mean(all_tokens_used):.2f}") 
    print("-"*50)      
    print(f"  - Retry Statistics:")
    print(f"    * Total Retry Attempts: {total_retries}")
    print(f"    * Trials with Retries: {trials_with_retries}/{len(all_results)} ({trials_with_retries/len(all_results)*100:.1f}%)")
    print(f"    * Average Retries per Trial: {np.mean(all_retry_attempts):.2f}")
    print(f"    * Failed Trials (after all retries): {len(failed_results)}")
    print("="*50)

    # Write aggregate results and config
    overall_result = {
        "config": {
            "module": cli_args.module,
            "model_name": cli_args.model_name,
            "noise_level": cli_args.noise,
            "equation_difficulty": cli_args.equation_difficulty,
            "law_version": law_version,
            "model_system": cli_args.model_system,
            "trials": num_trials,
            "max_retries": max_retries,
            "LLM judge": judge_model_name,
            "Agent backend": cli_args.agent_backend,
            "runtime_seconds": end_time - start_time
        },
        "aggregate": {
            # Metrics for all trials
            "all_trials": {
                "average_rmsle": float(np.nanmean(finite_all_rmsle)),
                "average_exact_accuracy": float(np.mean(all_accuracies)),
                "average_rounds": float(np.mean(all_turns)),
                "average_experiments": float(np.mean(all_experiments_used)),
                "average_total_tokens": float(np.mean(all_tokens_used)),
                "num_total_trials": len(all_results)
            },
            "retry_statistics": {
                "total_retry_attempts": total_retries,
                "trials_with_retries": trials_with_retries,
                "trials_with_retries_percentage": float(trials_with_retries/len(all_results)*100),
                "average_retries_per_trial": float(np.mean(all_retry_attempts)),
                "failed_trials_after_retries": len(failed_results)
            }
        },
    }
    agg_path = os.path.join(results_dir, "aggregated_results.json")
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(overall_result, f, indent=2)
        
    print(f"All results and logs for law_version '{law_version}' written to {results_dir}")
    print(f"--- Finished experiment for law_version: {law_version} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Scientific Discovery Benchmark Runner")
    parser.add_argument("--module", type=str, default="m0_gravity", help="Name of the module to test (e.g., m0_gravity).")
    parser.add_argument("--model_name", type=str, default="gpt41mini", help="Name of the LLM to use.")
    parser.add_argument("-n", "--noise", type=float, default=0.0, help="Noise level for experiments (e.g., 0, 0.01, 0.1).")
    parser.add_argument("-t", "--trials", type=int, default=12, help="Number of parallel trials to run.")
    parser.add_argument("-d", "--equation_difficulty", type=str, default="easy", choices=["easy", "medium", "hard"],
                      help="Difficulty level of the equation: easy, medium, or hard.")
    parser.add_argument("-m", "--model_system", type=str, default="vanilla_equation", choices=["vanilla_equation", "simple_system", "complex_system"],
                      help="Model system selected to test the agent: vanilla_equation, simple_system, complex_system")
    parser.add_argument("-l", "--law_version", type=str, default="all",
                      help="Specific law version to use, 'all' for all versions, or None for random selection or a specific version (e.g. v0, v1, v2)")
    parser.add_argument("-b", "--agent_backend", type=str, default="vanilla_agent", choices=["vanilla_agent", "code_assisted_agent"],
                      help="Agent backend to use for exploration. Default is vanilla_agent. When code_assisted_agent is selected, LLM is equipped with <python> tool use.")
    cli_args = parser.parse_args()

    # --- Pre-flight Check ---
    try:
        module = importlib.import_module(f"modules.{cli_args.module}")
        print(f"Successfully located module: {cli_args.module}")
    except ImportError:
        print(f"FATAL: Module '{cli_args.module}' not found in the 'modules/' directory.")
        print(f"Please ensure the file 'modules/{cli_args.module}' exists.")
        sys.exit(1)

    # Determine which law versions to run
    versions_to_run = []
    if cli_args.law_version == "all":
        if hasattr(module, 'get_available_law_versions'):
            versions_to_run = module.get_available_law_versions(cli_args.equation_difficulty)
            if not versions_to_run:
                print(f"FATAL: No available law versions found for equation difficulty '{cli_args.equation_difficulty}'.")
                sys.exit(1)
        else:
            print(f"FATAL: Module '{cli_args.module}' does not support law version validation")
            sys.exit(1)
    else:
        # Also check if the specified single version is valid
        if cli_args.law_version is not None and hasattr(module, 'get_available_law_versions'):
            available_versions = module.get_available_law_versions(cli_args.equation_difficulty)
            if cli_args.law_version not in available_versions:
                print(f"FATAL: Law version '{cli_args.law_version}' not found for equation difficulty '{cli_args.equation_difficulty}'.")
                print(f"Available versions for {cli_args.equation_difficulty}: {available_versions}")
                print("Please specify a valid law version, 'all', or use None for random selection.")
                sys.exit(1)
        versions_to_run = [cli_args.law_version]

    # Distribute trials among versions
    total_trials = cli_args.trials
    num_versions = len(versions_to_run)
    trials_per_version = {}
    if num_versions > 0:
        base_trials = total_trials // num_versions
        extra_trials = total_trials % num_versions
        for i, version in enumerate(versions_to_run):
            trials_per_version[version] = base_trials + (1 if i < extra_trials else 0)

    # Run a separate experiment for each version
    for version in versions_to_run:
        num_trials_for_version = trials_per_version.get(version, 0)
        if num_trials_for_version > 0:
            run_experiment_for_version(cli_args, module, version, num_trials_for_version)
        else:
            print(f"Skipping law_version {version} as it has 0 trials assigned.")