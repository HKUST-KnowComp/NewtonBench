import numpy as np
import math
from typing import Union, Dict, List, Any, Tuple
from utils.noise import inject_noise
import re
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .m3_types import (
    ABSOLUTE_POWER_PRECISION,
    ABSOLUTE_TEMPERATURE_PRECISION,
    ABSOLUTE_HEAT_FLUX_PRECISION,
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
    if not re.search(r'def\s+discovered_law\s*\(k,\s*A,\s*delta_T,\s*d\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_simple_fourier_experiment(
    k: float,
    A: float,
    delta_T: float,
    d: float,
    num_points: int = 20,
    noise_level: float = 0.01,
    force_law: callable = None
) -> Dict[str, List[str]]:
    """
    Simulate a 1D Fourier heat conduction experiment, tracking temperature profiles.
    
    Args:
        k (float): Thermal conductivity coefficient
        A (float): Cross-sectional area
        delta_T (float): Temperature difference
        d (float): Distance/thickness
        num_points (int): Number of spatial points to sample
        noise_level (float): Relative noise level for measurements
        force_law (callable): Function to compute the power law
    Returns:
        dict: Spatial data with keys 'x', 'T' (all as lists of strings)
    """
    if force_law is None:
        raise ValueError("force_law must be provided")
    
    # Compute power using the provided force law (ground truth law) with noisy inputs
    P = force_law(k, A, delta_T, d)
    
    # Generate positions (0 to d) using noisy distance
    x = np.linspace(0, d, num_points)
    # Temperature profile: Linear function of P
    temperatures = delta_T - (P / (k * A)) * x
    
    # No need to add noise to temperatures since inputs already have noise
    noisy_temperatures = inject_noise(temperatures, noise_level, ABSOLUTE_TEMPERATURE_PRECISION)
    
    # Convert to string format
    x_list = ["{:.6e}".format(float(pos)) for pos in x]
    temp_list = ["{:.6e}".format(float(temp)) for temp in noisy_temperatures.tolist()]
    
    return {'x': x_list, 'T': temp_list}

def _run_difficult_fourier_experiment(
    k: float,
    A: float,
    delta_T: float,
    d: float,
    num_points: int = 20,
    noise_level: float = 0.01,
    force_law: callable = None,
    difficulty: str = 'easy'
) -> Dict[str, List[str]]:
    """
    Simulate a 1D Fourier heat conduction experiment, tracking temperature profiles and heat flux.
    
    Args:
        k (float): Thermal conductivity coefficient
        A (float): Cross-sectional area
        delta_T (float): Temperature difference
        d (float): Distance/thickness
        num_points (int): Number of spatial points to sample
        noise_level (float): Relative noise level for measurements
        force_law (callable): Function to compute the power law
        difficulty (str): Difficulty level ('easy', 'medium', 'hard') for temperature profile
    Returns:
        dict: Spatial data with keys 'x', 'heat_flux' (all as lists of strings)
    """
    if force_law is None:
        raise ValueError("force_law must be provided")

    # Compute power using the provided force law (ground truth law) with noisy inputs
    P = force_law(k, A, delta_T, d)
    
    # Generate positions (0 to d) using noisy distance
    x = np.linspace(0, d, num_points)
    
    # Temperature profile: Different profiles based on difficulty level
    if difficulty == 'easy':
        # For Easy Ground Truth Law
        temperatures = delta_T - np.exp(-x * P / (k * A * delta_T))
    elif difficulty == 'medium':
        # For Medium Ground Truth Law
        temperatures = delta_T * np.exp(-x * P / (k * (A + delta_T)))
    elif difficulty == 'hard':
        # For Hard Ground Truth Law
        temperatures = delta_T * np.exp(-x * P / (k * (A + delta_T) ** 2.5))
    else:
        # Fallback to simple case
        temperatures = delta_T - np.exp(-x * P / (k * A * delta_T))
    
    # Calculate heat flux using Fourier's law: q = -k * dT/dx
    q = -k * np.gradient(temperatures, x)  # Finite difference approximation
    
    noisy_heat_flux = inject_noise(q, noise_level, ABSOLUTE_HEAT_FLUX_PRECISION)

    # Convert to string format
    x_list = ["{:.6e}".format(float(pos)) for pos in x]
    flux_list = ["{:.6e}".format(float(flux)) for flux in noisy_heat_flux.tolist()]
    
    return {
        'x': x_list, 
        'heat_flux': flux_list
    }

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system, and complex_system modes for Fourier's law.
    Args:
        k: Thermal conductivity coefficient (can also be passed via kwargs)
        A: Cross-sectional area (can also be passed via kwargs)
        delta_T: Temperature difference (can also be passed via kwargs)
        d: Distance/thickness (can also be passed via kwargs)
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameters
    Returns:
        For vanilla_equation: power measurement (float)
        For simple/complex_system: time series data (dict)
    """
    # Handle flexible parameter passing - using module 8 approach
    k = kwargs.get('k', 1.0)
    A = kwargs.get('A', 1.0)
    delta_T = kwargs.get('delta_T', 1.0)
    d = kwargs.get('d', 1.0)
    
    force_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:        
        # Calculate result with noisy inputs
        true_power = force_law(k, A, delta_T, d)
        return inject_noise(true_power, noise_level, ABSOLUTE_POWER_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        return _run_simple_fourier_experiment(
            k=k,
            A=A,
            delta_T=delta_T,
            d=d,
            num_points=kwargs.get('num_points', 20),
            noise_level=noise_level,
            force_law=force_law
        )
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        return _run_difficult_fourier_experiment(
            k=k,
            A=A,
            delta_T=delta_T,
            d=d,
            num_points=kwargs.get('num_points', 20),
            noise_level=noise_level,
            force_law=force_law,
            difficulty=difficulty
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
    k_values = np.exp(np.random.uniform(np.log(1e-1), np.log(1e1), num_points))  # Log uniform 10^-1 to 10^1
    A_values = np.exp(np.random.uniform(np.log(1e-4), np.log(1e-2), num_points))  # Log uniform 10^-4 to 10^-2
    delta_T_values = np.exp(np.random.uniform(np.log(1e1), np.log(1e3), num_points))  # Log uniform 10^1 to 10^3
    d_values = np.exp(np.random.uniform(np.log(1e-2), np.log(1e0), num_points))  # Log uniform 10^-2 to 10^0
    
    test_data = {
        'k': k_values,
        'A': A_values,
        'delta_T': delta_T_values,
        'd': d_values,
    }
    
    # Define parameter mapping for Fourier law module
    parameter_mapping = {
        "k": "k",
        "A": "A", 
        "delta_T": "delta_T",
        "d": "d"
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
