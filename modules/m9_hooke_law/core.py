import numpy as np
import re
from typing import Dict, List, Union, Any, Tuple
from modules.common.types import ExperimentSystem
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from utils.noise import inject_noise
from .m9_types import (
    ABSOLUTE_ENERGY_PRECISION,
    ABSOLUTE_VELOCITY_PRECISION,
    HOOKE_DEFAULTS,
    ENERGY_DEFAULTS
)
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
    if not re.search(r'def\s+discovered_law\s*\(k,\s*x\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None



def calculate_exponential_energy_loss(x: float, x_scale: float = None) -> float:
    """
    Calculate exponential energy loss factor based on displacement.
    
    Physics: Energy loss increases exponentially with displacement due to:
    - Material fatigue and stress accumulation
    - Geometric nonlinearities and buckling effects
    - Microstructural damage and crack propagation
    
    Args:
        x: Displacement in meters
        x_scale: Characteristic displacement scale for energy decay
    
    Returns:
        Energy retention factor between 0 and 1 (exponential decay)
    """
    if x_scale is None:
        x_scale = HOOKE_DEFAULTS['default_displacement_scale']
    
    # Exponential decay: energy retention = exp(-x / x_scale)
    # Small x: energy retention ≈ 1 (minimal loss)
    # Large x: energy retention ≈ 0 (significant loss)
    energy_retention = np.exp(-x / x_scale)
    
    return energy_retention

def _run_difficult_hooke_velocity_experiment(
    k: float,
    x: float,
    m: float,
    noise_level: float = 0.01,
    energy_law: callable = None
) -> float:
    """
    Simulate a difficult Hooke's law experiment to calculate realistic maximum velocity.
    
    Args:
        k (float): Spring constant in N/m
        x (float): Displacement from equilibrium in meters
        m (float): Mass in kg
        x_scale (float): Displacement scale for friction calculation (uses default if None)
        noise_level (float): Relative noise level for measurements
        energy_law (callable): Function to compute the energy law
    Returns:
        float: Realistic maximum velocity in m/s
    """
    if energy_law is None:
        raise ValueError("energy_law must be provided")
    
    # Step 1: Calculate elastic potential energy using the ground truth law
    U = energy_law(k, x)
    
    # Step 2: Calculate theoretical maximum velocity using energy conservation
    # U = 1/2 * m * v_max^2, so v_max = sqrt(2*U/m)
    v_max = np.sqrt(2 * U / m)
    
    # Step 3: Calculate realistic maximum velocity (accounting for exponential energy loss)
    # Use fixed default x_scale for consistent exponential energy loss behavior
    energy_retention = calculate_exponential_energy_loss(x, HOOKE_DEFAULTS['default_displacement_scale'])
    real_v_max = energy_retention * v_max
    
    # Step 4: Return realistic maximum velocity
    return inject_noise(real_v_max, noise_level, ABSOLUTE_VELOCITY_PRECISION)

def _run_simple_hooke_velocity_experiment(
    k: float,
    x: float,
    m: float,
    noise_level: float = 0.01,
    energy_law: callable = None
) -> float:
    """
    Simulate a simple Hooke's law experiment to calculate net kinetic energy after air resistance.
    
    Args:
        k (float): Spring constant in N/m
        x (float): Displacement from equilibrium in meters
        m (float): Mass in kg
        noise_level (float): Relative noise level for measurements
        energy_law (callable): Function to compute the energy law
    Returns:
        float: Net kinetic energy after air resistance in J
    """
    if energy_law is None:
        raise ValueError("energy_law must be provided")

    # Air resistance coefficient
    k_air = 0.2
    
    # Early validation for invalid inputs
    if m <= 0:
        return float('nan')
    
    # Calculate elastic potential energy using the ground truth law
    U = energy_law(k, x)
    
    # Calculate maximum velocity using energy conservation: U = 1/2 * m * v_max^2
    # v_max = sqrt(2*U/m)
    v_max = np.sqrt(2 * U / m)
    
    # Calculate kinetic energy without air resistance
    KE = 0.5 * m * (v_max ** 2)
    
    # Calculate kinetic energy loss due to air resistance
    # KE_loss = -k_air * x * v_max^2
    KE_loss = -k_air * x * (v_max ** 2)
    
    # Return net kinetic energy after air resistance
    return inject_noise(KE - KE_loss, noise_level, ABSOLUTE_ENERGY_PRECISION)

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system, and complex_system modes for Hooke's law.
    
    Args:
        k: Spring constant in N/m (can also be passed via kwargs) - for vanilla_equation/simple_system
        x: Displacement from equilibrium in meters (can also be passed via kwargs) - for vanilla_equation/simple_system
        t: Time elapsed in seconds (can also be passed via kwargs) - for vanilla_equation/simple_system
        k1: Spring constant of first spring in N/m (can also be passed via kwargs) - for complex_system
        k2: Spring constant of second spring in N/m (can also be passed via kwargs) - for complex_system
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameters
        
    Returns:
        For vanilla_equation: energy measurement (float)
        For simple_system: realistic maximum velocity (float)
        For complex_system: net kinetic energy after air resistance (float)
    """
    # Handle flexible parameter passing - using module 8 approach
    k = kwargs.get('k', 1.0)
    x = kwargs.get('x', 1.0)
    t = kwargs.get('t', 1.0)
    m = kwargs.get('m', HOOKE_DEFAULTS['default_mass'])
    x_scale = kwargs.get('x_scale', HOOKE_DEFAULTS['default_displacement_scale'])  # NEW
    k1 = kwargs.get('k1', 1.0)
    k2 = kwargs.get('k2', 1.0)
    
    # Get the ground truth law
    energy_law, _ = get_ground_truth_law(difficulty, law_version)
    
    if system == ExperimentSystem.VANILLA_EQUATION:
        # Vanilla equation
        energy = energy_law(k, x)
        return inject_noise(energy, noise_level, ABSOLUTE_ENERGY_PRECISION)
    
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        # Simple system: calculate net kinetic energy after air resistance
        m = kwargs.get('m', HOOKE_DEFAULTS['default_mass'])
        return _run_simple_hooke_velocity_experiment(
            k, x, m, noise_level, energy_law
        )
    
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        # Complex system: calculate realistic maximum velocity
        m = kwargs.get('m', HOOKE_DEFAULTS['default_mass'])
        return _run_difficult_hooke_velocity_experiment(
            k, x, m, noise_level, energy_law
        )
    
    else:
        raise ValueError(f"Invalid system: {system}. Choose from 'vanilla_equation', 'simple_system', 'complex_system'")

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
        law_version: Specific law version ('v0') or None for default selection
        judge_model_name: Name of the LLM model to use for symbolic equivalence checking
        trial_info: Optional trial information dictionary
    Returns:
        Dictionary containing evaluation results
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
    k_values = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3), num_points))  # Log uniform 0.001 to 1000 N/m
    x_values = np.exp(np.random.uniform(np.log(1e-3), np.log(1e0), num_points))  # Log uniform 0.001 to 1 m
    
    test_data = {
        'k': k_values,
        'x': x_values,
    }
    
    # Define parameter mapping for Hooke's law module
    parameter_mapping = {
        "k": "k",
        "x": "x"
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
