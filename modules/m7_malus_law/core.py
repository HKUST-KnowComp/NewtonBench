"""
Core experiment logic and evaluation for Module 7: Malus's Law

This module contains the main experiment runner and evaluation functions
for discovering the relationship between light intensity and polarization angle.
"""

import numpy as np
from typing import Dict, Any, Union, List, Tuple
import ast
import re
import numpy as np
from utils.noise import inject_noise
from modules.common.types import ExperimentSystem
from modules.common.evaluation import evaluate_law as shared_evaluate_law

from .m7_types import (
    MALUS_DEFAULTS, ABSOLUTE_INTENSITY_PRECISION, ABSOLUTE_RATIO_PRECISION
)
from .laws import get_ground_truth_law
from .physics import (
    calculate_transmitted_intensity, calculate_intensity_at_angle
)

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """
    Validate the LLM's function definition.
    Args:
        code: The complete function string
    Returns:
        (is_valid, error_message)
    """
    # Check function name and signature
    if not re.search(r'def\s+discovered_law\s*\(I_0,\s*theta\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_simple_malus_experiment(
    I_0: float,
    theta: float,
    noise_level: float = 0.01,
    malus_law: callable = None
) -> float:
    """
    Simulate a simple Malus's Law experiment that outputs intensity ratio.
    
    Args:
        I_0: Initial light intensity in W/m²
        theta: Angle between polarization direction and polarizer axis in radians
        noise_level: Relative noise level for measurements
        malus_law: Function to compute the Malus's Law (ground truth law for specified difficulty)
    
    Returns:
        Intensity ratio I/I_0 (dimensionless)
    """
    if malus_law is None:
        raise ValueError("malus_law must be provided")
    
    # Simple parameter validation
    if I_0 <= 0:
        return float("nan")
    if theta <= 0 or theta > np.pi/2:
        return float("nan")
    
    # Calculate transmitted intensity using ground truth law of specified difficulty
    I = malus_law(I_0, theta)
    ratio = inject_noise(I / I_0, noise_level, ABSOLUTE_RATIO_PRECISION)
    
    # Return intensity ratio instead of absolute intensity
    return ratio

def _run_difficult_malus_experiment(
    I_0: float,
    theta: float,
    num_points: int = 20,
    noise_level: float = 0.01,
    malus_law: callable = None
) -> float:
    """
    Simulate a complex two-polarizer Malus's Law experiment.
    
    The system consists of three polarizers:
    - Polarizer 1 → Polarizer 2: I = ground_truth_law(I_0, theta)
    - Polarizer 2 → Polarizer 3: I_1 = ground_truth_law(I, theta)
    - Output: I_1 - I_0 (intensity difference)
    
    Args:
        I_0: Initial light intensity in W/m²
        theta: Angle between polarization direction and polarizer axis in radians
        num_points: Number of angle points to sample (unused, kept for compatibility)
        noise_level: Relative noise level for measurements
        malus_law: Function to compute the Malus's Law (ground truth law for specified difficulty)
    
    Returns:
        Intensity difference I_1 - I_0 (float)
    """
    if malus_law is None:
        raise ValueError("malus_law must be provided")
    
    # Simple parameter validation
    if I_0 <= 0:
        return float("nan")
    if theta <= 0 or theta > np.pi/2:
        return float("nan")
    
    # First polarizer: I_0 → I
    I = malus_law(I_0, theta)
    
    # Second polarizer: I → I_1
    I_1 = malus_law(I, theta)
    
    # Calculate intensity difference: I_1 - I_0
    intensity_difference = I_1 - I_0
    
    return inject_noise(intensity_difference, noise_level, ABSOLUTE_INTENSITY_PRECISION)

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system, and complex_system modes for Malus's Law.
    
    Args:
        I_0: Initial light intensity in W/m² (can also be passed via kwargs)
        theta: Angle between polarization direction and polarizer axis in radians (can also be passed via kwargs)
        num_points: Number of angle points for complex system (can also be passed via kwargs)
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        law_version: Specific version of the law (currently unused, for future compatibility)
        **kwargs: Additional parameters
        
    Returns:
        For vanilla_equation: transmitted intensity measurement (float) in W/m²
        For simple_system: intensity ratio (float) - dimensionless
        For complex_system: intensity difference (float) in W/m²
    """
    # Handle flexible parameter passing - using module 8 approach
    I_0 = kwargs.get('I_0', MALUS_DEFAULTS['default_initial_intensity'])
    theta = kwargs.get('theta', MALUS_DEFAULTS['default_angle'])
    num_points = kwargs.get('num_points', MALUS_DEFAULTS['num_points'])
    
    # Get the ground truth law
    malus_law, _ = get_ground_truth_law(difficulty, law_version)
    
    if system == ExperimentSystem.VANILLA_EQUATION:
        true_intensity = malus_law(I_0, theta)
        return inject_noise(true_intensity, noise_level, ABSOLUTE_INTENSITY_PRECISION)
    
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        # Simple system: basic Malus's Law experiment
        return _run_simple_malus_experiment(I_0, theta, noise_level, malus_law)
    
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        # Complex system: complex polarization system with angle series
        return _run_difficult_malus_experiment(I_0, theta, num_points, noise_level, malus_law)
    
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
        llm_function_str: String representation of the LLM's function
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        law_version: Specific version of the law (currently unused, for future compatibility)
        judge_model_name: Name of the judge model for evaluation
        trial_info: Additional trial information (currently unused)
        
    Returns:
        Dictionary containing evaluation results
    """
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}
    
    # Get the ground truth law
    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    
    # Generate test data
    num_points = 5000
    # Use log-uniform sampling for all parameters
    I_0_values = np.exp(np.random.uniform(np.log(100.0), np.log(2000.0), num_points))  # Log uniform 100 to 2000
    theta_values = np.random.uniform(1e-6, np.pi/2, num_points)  # Uniform greater than 0 to π/2
    
    test_data = {
        'I_0': I_0_values,
        'theta': theta_values,
    }
    
    # Define parameter mapping for Malus's Law module
    parameter_mapping = {
        "I_0": "I_0",
        "theta": "theta"
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
