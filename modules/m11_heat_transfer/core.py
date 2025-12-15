import numpy as np
import re
from typing import Dict, List, Union, Any, Tuple
from modules.common.types import ExperimentSystem
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from utils.noise import inject_noise
from .m11_types import (
    ABSOLUTE_POWER_PRECISION,
    ABSOLUTE_HEAT_TRANSFER_PRECISION,
    HEAT_TRANSFER_CONSTANTS
)
from .laws import get_ground_truth_law
import pandas as pd

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """
    Validate the LLM's function definition.
    Args:
        code: The complete function string
    Returns:
        (is_valid, error_message)
    """
    # Check function name and signature
    if not re.search(r'def\s+discovered_law\s*\(m,\s*c,\s*delta_T\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def generate_random_c_constant_factor() -> float:
    """
    Generate random c_constant factor between 0.7 and 1.2.
     
    Returns:
        Random factor between 0.7 and 1.2
    """
    from modules.m11_heat_transfer.m11_types import HEAT_TRANSFER_CONSTANTS
    
    # Generate random factor between min and max bounds
    factor = np.random.uniform(
        HEAT_TRANSFER_CONSTANTS['MIN_FACTOR'],
        HEAT_TRANSFER_CONSTANTS['MAX_FACTOR']
    )
    
    return factor

def calculate_energy_loss_before_distribution() -> float:
    """
    Calculate random energy loss before power distribution.
        
    Returns:
        Energy loss fraction between 0.18 and 0.22 (18-22%)
    """
    from modules.m11_heat_transfer.m11_types import HEAT_TRANSFER_CONSTANTS
    
    # Generate random energy loss between 18-22%
    energy_loss = np.random.uniform(
        HEAT_TRANSFER_CONSTANTS['MIN_ENERGY_LOSS'],
        HEAT_TRANSFER_CONSTANTS['MAX_ENERGY_LOSS']
    )
    
    return energy_loss

def generate_heat_transfer_distribution() -> tuple[float, float, float]:
    """
    Generate random heat transfer distribution among conduction, convection, and radiation.
        
    Returns:
        Tuple of (f_cond, f_conv, f_rad) where f_cond + f_conv + f_rad = 1.0
    """
    from modules.m11_heat_transfer.m11_types import HEAT_TRANSFER_CONSTANTS
    
    # Generate three random fractions between min and max
    f_cond = np.random.uniform(HEAT_TRANSFER_CONSTANTS['MIN_DISTRIBUTION'], 
                               HEAT_TRANSFER_CONSTANTS['MAX_DISTRIBUTION'])
    f_conv = np.random.uniform(HEAT_TRANSFER_CONSTANTS['MIN_DISTRIBUTION'], 
                               HEAT_TRANSFER_CONSTANTS['MAX_DISTRIBUTION'])
    f_rad = np.random.uniform(HEAT_TRANSFER_CONSTANTS['MIN_DISTRIBUTION'], 
                              HEAT_TRANSFER_CONSTANTS['MAX_DISTRIBUTION'])
    
    # Normalize to ensure they sum to 1.0
    total = f_cond + f_conv + f_rad
    f_cond /= total
    f_conv /= total
    f_rad /= total
    
    return f_cond, f_conv, f_rad

def _run_simple_heat_transfer_power_experiment(
    m: float,
    c: float,
    delta_T: float,
    noise_level: float = 0.01,
    heat_law: callable = None
) -> Dict[str, float]:
    """
    Simulate heat transfer power distribution among conduction, convection, and radiation.
    """
    if heat_law is None:
        raise ValueError("heat_law must be provided")

    Q_total = heat_law(m, c, delta_T)

    t = (m * c) / HEAT_TRANSFER_CONSTANTS['TIME_SCALING_FACTOR']
    
    P_total = Q_total / t

    energy_loss_fraction = calculate_energy_loss_before_distribution()
    available_power = P_total * (1 - energy_loss_fraction)  
    
    f_cond, f_conv, f_rad = generate_heat_transfer_distribution()
    
    P_cond = inject_noise(f_cond * available_power, noise_level, ABSOLUTE_POWER_PRECISION)
    P_conv = inject_noise(f_conv * available_power, noise_level, ABSOLUTE_POWER_PRECISION)
    P_rad = inject_noise(f_rad * available_power, noise_level, ABSOLUTE_POWER_PRECISION)
    
    return {
        'P_cond': P_cond,
        'P_conv': P_conv,
        'P_rad': P_rad
    }


def calculate_light_bulb_power_from_heat_difference(
    m: float,
    c: float,
    delta_T: float,
    heat_law: callable,
    noise_level: float = 0.01
) -> int:
    """
    Calculate number of light bulbs that can be powered from heat transfer difference.
    """
    from modules.m11_heat_transfer.m11_types import HEAT_TRANSFER_CONSTANTS
        
    Q_original = heat_law(m, c, delta_T)
    
    random_factor = generate_random_c_constant_factor()
    c_alternative = c * random_factor
    Q_alternative = heat_law(m, c_alternative, delta_T)
    
    Q_difference = abs(Q_original - Q_alternative)
    
    t = (m * c) / HEAT_TRANSFER_CONSTANTS['TIME_SCALING_FACTOR']
    
    P_total = Q_difference / t

    energy_loss_fraction = calculate_energy_loss_before_distribution()
    available_power = P_total * (1 - energy_loss_fraction)
    
    light_bulb_power = HEAT_TRANSFER_CONSTANTS['LIGHT_BULB_POWER']
    
    if pd.isna(available_power) or pd.isna(light_bulb_power) or light_bulb_power == 0:
        return 0  
    
    available_power = inject_noise(available_power, noise_level, ABSOLUTE_POWER_PRECISION)
    num_light_bulbs = int(available_power / light_bulb_power)
    
    return num_light_bulbs

def _run_difficult_heat_transfer_light_bulb_experiment(
    m: float,
    c: float,
    delta_T: float,
    noise_level: float = 0.01,
    heat_law: callable = None
) -> int:
    """
    Simulate a difficult heat transfer experiment using light bulb power system.
    """
    if heat_law is None:
        raise ValueError("heat_law must be provided")
    
    return calculate_light_bulb_power_from_heat_difference(
        m, c, delta_T, heat_law, noise_level
    )

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system, and complex_system modes for heat transfer.
    
    Args:
        m: Mass (can also be passed via kwargs) - for vanilla_equation/simple_system
        c: c_constant (can also be passed via kwargs) - for vanilla_equation/simple_system
        delta_T: Temperature change (can also be passed via kwargs) - for vanilla_equation/simple_system
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameters
        
    Returns:
        For vanilla_equation: heat transfer measurement (float)
        For simple_system: power distribution dict {'P_cond': float, 'P_conv': float, 'P_rad': float}
                      (components sum to 78-82% of total power due to energy loss)
        For complex_system: number of light bulbs that can be powered (integer)
    """
    # Handle flexible parameter passing - using module 8 approach
    m = kwargs.get('m', 1.0)
    c = kwargs.get('c', 1000.0)
    delta_T = kwargs.get('delta_T', 1.0)
    
    # Get the ground truth law
    heat_law, _ = get_ground_truth_law(difficulty, law_version)
    
    if system == ExperimentSystem.VANILLA_EQUATION:
        # Vanilla equation
        heat_transfer = heat_law(m, c, delta_T)
        return inject_noise(heat_transfer, noise_level, ABSOLUTE_HEAT_TRANSFER_PRECISION)
    
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        # Simple system: heat transfer power distribution among conduction, convection, and radiation
        return _run_simple_heat_transfer_power_experiment(
            m, c, delta_T, noise_level, heat_law
        )
    
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        # Complex system: light bulb power system using heat transfer differences
        return _run_difficult_heat_transfer_light_bulb_experiment(
            m, c, delta_T, noise_level, heat_law
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
    m_values = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3), num_points))  # Log uniform 0.001 to 1000 kg
    c_values = np.exp(np.random.uniform(np.log(1e2), np.log(1e4), num_points))   # Log uniform 100 to 10000 J/(kg·K)
    delta_T_values = np.exp(np.random.uniform(np.log(1e1), np.log(1e3), num_points))  # Log uniform 0.01 to 100 K
    
    test_data = {
        'm': m_values,
        'c': c_values,
        'delta_T': delta_T_values,
    }
    
    # Define parameter mapping for heat transfer module
    parameter_mapping = {
        "m": "m",
        "c": "c",
        "delta_T": "delta_T"
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
