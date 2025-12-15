import numpy as np
import math
from typing import Union, Dict, List, Any, Tuple
from utils.noise import inject_noise
import re
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .m5_types import (
    ABSOLUTE_ACTIVITY_PRECISION,
    ABSOLUTE_RATIO_PRECISION,
    TWO_DIM_DEFAULTS,
    LINEAR_DEFAULTS
)
from modules.common.types import ExperimentSystem
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """
    Validate the LLM's function definition.
    Args:
        code: The complete function string
    Returns:
        (is_valid, error_message)
    """
    # Check function name and signature
    if not re.search(r'def\s+discovered_law\s*\(N0,\s*lambda_constant,\s*t\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_difficult_radioactive_decay_experiment(
    N0a: float,
    N0b: float,
    lambda_a: float,
    lambda_b: float,
    t: float,
    num_points: int = 20,
    noise_level: float = 0.01,
    decay_law: callable = None
) -> Dict[str, List[str]]:
    """
    Simulate a complex radioactive decay experiment with two isotopes, tracking their ratio over time.
    """
    if decay_law is None:
        raise ValueError("decay_law must be provided")
    
    time_points = np.linspace(0, t, num_points)
    
    ratios = []
    for time_point in time_points:
        Na_t = decay_law(N0a, lambda_a, time_point)  
        Nb_t = decay_law(N0b, lambda_b, time_point)
        
        Na_t = float(Na_t)
        Nb_t = float(Nb_t)
        

        if abs(Nb_t) < 1e-10:  
            ratio_t = np.nan  
        else:
            ratio_t = Na_t / Nb_t
            if not np.isfinite(ratio_t):
                ratio_t = np.nan 
        
        ratios.append(ratio_t)
    ratios = np.array(ratios)
    ratios = inject_noise(ratios, noise_level, ABSOLUTE_RATIO_PRECISION)
    time_list = ["{:.6e}".format(float(time_point)) for time_point in time_points]
    ratio_list = ["{:.6e}".format(float(r)) for r in ratios]
    
    return {'time': time_list, 'ratio': ratio_list}

def _run_simple_radioactive_decay_experiment(
    N0: float,
    lambda_constant: float,
    t: float,
    num_points: int = 20,
    noise_level: float = 0.01,
    decay_law: callable = None
) -> Dict[str, List[str]]:
    """
    Simulate a radioactive decay experiment with a radiation detector, measuring activity over time.
    """
    if decay_law is None:
        raise ValueError("decay_law must be provided")
    
    time_points = np.linspace(0, t, num_points)
    
    measured_activities = []
    for time_point in time_points:
        remaining_atoms = decay_law(N0, lambda_constant, time_point)
            
        remaining_atoms = float(remaining_atoms)
        
        measured_activity = 0.7 * remaining_atoms
        measured_activities.append(measured_activity)
    
    measured_activities = np.array(measured_activities)
    measured_activities = inject_noise(measured_activities, noise_level, ABSOLUTE_ACTIVITY_PRECISION)
    time_list = ["{:.6e}".format(float(time_point)) for time_point in time_points]
    activity_list = ["{:.6e}".format(float(activity)) for activity in measured_activities]
    
    return {'time': time_list, 'measured_activity': activity_list}

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system, and complex_system modes for radioactive decay.
    Args:
        N0: Initial activity (can also be passed via kwargs) - for vanilla_equation
        lambda_constant: lambda constant (can also be passed via kwargs) - for vanilla_equation
        t: Time elapsed (can also be passed via kwargs)
        N0: Initial number of parent isotope atoms (can also be passed via kwargs) - for simple_system
        lambda_constant: lambda constant of parent isotope (can also be passed via kwargs) - for simple_system
        N0a: Initial number of nuclei for Isotope A (can also be passed via kwargs) - for complex_system
        N0b: Initial number of nuclei for Isotope B (can also be passed via kwargs) - for complex_system
        lambda_a: lambda constant for Isotope A (can also be passed via kwargs) - for complex_system
        lambda_b: lambda constant for Isotope B (can also be passed via kwargs) - for complex_system
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameter
    Returns:
        For vanilla_equation: activity measurement (float)
        For simple_system: temporal measured activity data (dict)
        For complex_system: temporal ratio data (dict)
    """
    # Handle flexible parameter passing - using module 8 approach
    N0 = kwargs.get('N0', 1.0)
    lambda_constant = kwargs.get('lambda_constant', 1.0)
    t = kwargs.get('t', 1.0)
    N0a = kwargs.get('N0a', 1.0)
    N0b = kwargs.get('N0b', 1.0)
    lambda_a = kwargs.get('lambda_a', 1.0)
    lambda_b = kwargs.get('lambda_b', 1.0)
    
    decay_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        true_activity = decay_law(N0, lambda_constant, t)
        return inject_noise(true_activity, noise_level, ABSOLUTE_ACTIVITY_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        return _run_simple_radioactive_decay_experiment(
            N0=N0,
            lambda_constant=lambda_constant,
            t=t,
            num_points=kwargs.get('num_points', 20),
            noise_level=noise_level,
            decay_law=decay_law
        )
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        return _run_difficult_radioactive_decay_experiment(
            N0a=N0a,
            N0b=N0b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            t=t,
            num_points=kwargs.get('num_points', 20),
            noise_level=noise_level,
            decay_law=decay_law
        )
    else:
        raise ValueError(f"Invalid system: {system}. Must be one of {[e.value for e in ExperimentSystem]}")

def evaluate_law(
    llm_function_str: str,
    param_description: str,
    difficulty: str = 'easy',
    law_version: str = None,
    judge_model_name: str = "nemotron-ultra",
    trial_info=None,
) -> dict:
    """
    Evaluator assessing the symbolic equivalence and RMSLE of the LLM's submitted function.
    Args:
        llm_function_str: The submitted Python function as a string
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        judge_model_name: Name of the LLM model to use for symbolic equivalence checking
        trial_info: Optional trial information dictionary
    Returns:
        Dictionary containing evaluation metrics
    """
    # Validate LLM function
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {
            "rmsle": float('nan'),
            "exact_accuracy": 0.0,
            "symbolic_equivalent": False,
            "symbolic_msg": validation_error,
            "error": validation_error
        }
    
    # --- Extract ground truth law and test data ---
    gt_law, selected_law_version = get_ground_truth_law(difficulty, law_version)
    
    # Generate test data
    num_points = 5000
    # Use log-uniform sampling for all parameters
    N0_values = np.exp(np.random.uniform(np.log(1e0), np.log(1e2), num_points))  # Log uniform 1 to 100
    lambda_constant_values = np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), num_points))  # Log uniform 10^-3 to 10^-1
    t_values = np.exp(np.random.uniform(np.log(1e-2), np.log(1e1), num_points))  # Log uniform 10^-2 to 10^1
    
    test_data = {
        'N0': N0_values,
        'lambda_constant': lambda_constant_values,
        't': t_values,
    }
    
    # Define parameter mapping for radioactive decay module
    parameter_mapping = {
        "N0": "N0",
        "lambda_constant": "lambda_constant", 
        "t": "t"
    }
    
    return shared_evaluate_law(
        llm_function_str=llm_function_str,
        gt_law=gt_law,
        test_data=test_data,
        parameter_mapping=parameter_mapping,
        param_description=param_description,
        judge_model_name=judge_model_name,
        trial_info=trial_info,
        symbolic_check=True
    )

