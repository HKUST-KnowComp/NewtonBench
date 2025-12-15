import numpy as np
import math
from typing import Union, Dict, List, Any, Tuple
from utils.noise import inject_noise
import re
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .m1_types import (
    ExperimentSystem,
    ABSOLUTE_FORCE_PRECISION,
    ABSOLUTE_ENERGY_PRECISION,
    ABSOLUTE_VELOCITY_PRECISION,
    TWO_DIM_DEFAULTS,
    LINEAR_DEFAULTS
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
    if not re.search(r'def\s+discovered_law\s*\(q1,\s*q2,\s*distance\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_linear_coulomb_experiment(
    q1: float,
    m1: float,  # Mass of q1
    q2: float,
    m2: float,  # Mass of q2
    distance: float,
    duration: float,
    time_step: float,
    noise_level: float = 0.01,
    force_law: callable = None
) -> Dict[str, List[str]]:
    """
    Simulate a 1D Coulomb experiment with F=ma dynamics, tracking velocity.
    q1 and q2 decay at different rates (q1: 10, q2: 5).
    """
    if force_law is None:
        raise ValueError("force_law must be provided")

    decay_rate_q1 = 10.0
    decay_rate_q2 = 5.0
    
    num_steps = int(duration / time_step)
    if num_steps <= 0:
        return {'time': [], 'position': [], 'velocity': []}

    times = np.arange(num_steps) * time_step
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)

    positions[0] = distance
    velocities[0] = 0.0  
    
    for i, t in enumerate(times):
        positions[i] = positions[i-1] if i > 0 else distance
        velocities[i] = velocities[i-1] if i > 0 else 0.0
        
        q1_t = q1 * np.exp(-t / decay_rate_q1)
        q2_t = q2 * np.exp(-t / decay_rate_q2)
        
        current_distance = abs(positions[i])
        F_magnitude = force_law(abs(q1_t), abs(q2_t), current_distance)
        
        F_direction = 1 if q1_t * q2_t > 0 else -1
        F_on_q2 = F_magnitude * F_direction
        
        a = F_on_q2 / m2
        
        if i > 0:
            velocities[i] = velocities[i-1] + a * time_step
            positions[i] = positions[i-1] + velocities[i] * time_step

    noisy_velocities = inject_noise(velocities, noise_level, ABSOLUTE_VELOCITY_PRECISION)

    max_points = 20
    if len(times) > max_points:
        times = times[:max_points]
        noisy_velocities = noisy_velocities[:max_points]

    time_list = ["{:.3e}".format(float(t)) for t in times.tolist()]
    velocity_list = ["{:.6e}".format(float(v)) for v in noisy_velocities.tolist()]
    
    return {'time': time_list, 'velocity': velocity_list}

def _run_linear_coulomb_experiment_with_kinetic_energy(
    q1: float,
    m1: float,  # Mass of q1
    q2: float,
    m2: float,  # Mass of q2
    distance: float,
    duration: float,
    time_step: float,
    noise_level: float = 0.01,
    force_law: callable = None
) -> Dict[str, List[str]]:
    """
    Simulate a 1D Coulomb experiment with F=ma dynamics and kinetic energy tracking.
    q1 and q2 decay at different rates (q1: 10, q2: 5).
    """
    if force_law is None:
        raise ValueError("force_law must be provided")

    decay_rate_q1 = 10.0
    decay_rate_q2 = 5.0
    
    num_steps = int(duration / time_step)
    if num_steps <= 0:
        return {'time': [], 'position': [], 'velocity': [], 'kinetic_energy': []}

    times = np.arange(num_steps) * time_step
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    kinetic_energies = np.zeros(num_steps)

    positions[0] = distance
    velocities[0] = 0.0  
    
    for i, t in enumerate(times):
        positions[i] = positions[i-1] if i > 0 else distance
        velocities[i] = velocities[i-1] if i > 0 else 0.0
        
        q1_t = q1 * np.exp(-t / decay_rate_q1)
        q2_t = q2 * np.exp(-t / decay_rate_q2)
        
        current_distance = abs(positions[i])
        F_magnitude = force_law(abs(q1_t), abs(q2_t), current_distance)
        
        F_direction = 1 if q1_t * q2_t > 0 else -1
        F_on_q2 = F_magnitude * F_direction
        
        a = F_on_q2 / m2
        
        if i > 0:
            velocities[i] = velocities[i-1] + a * time_step
            positions[i] = positions[i-1] + velocities[i] * time_step
        
        kinetic_energies[i] = 0.5 * m2 * (velocities[i] ** 2)

    noisy_kinetic_energies = inject_noise(kinetic_energies, noise_level, ABSOLUTE_ENERGY_PRECISION)

    max_points = 20
    if len(times) > max_points:
        times = times[:max_points]
        noisy_kinetic_energies = noisy_kinetic_energies[:max_points]

    time_list = ["{:.3e}".format(float(t)) for t in times.tolist()]
    ke_list = ["{:.6e}".format(float(x)) for x in noisy_kinetic_energies.tolist()]
    
    return {'time': time_list, 'kinetic_energy': ke_list}

def run_experiment_for_module(
    noise_level: float = 0.01,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system (1D), and complex_system (2D) modes for Coulomb's law.
    Args:
        q1: Charge of first object (can also be passed via kwargs)
        q2: Charge of second object (can also be passed via kwargs)
        distance: Distance between objects (can also be passed via kwargs)
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameters
    Returns:
        For vanilla_equation: force measurement (float)
        For simple/complex_system: time series data (dict)
    """
    # Handle flexible parameter passing - using module 8 approach
    q1 = kwargs.get('q1', 1.0)
    q2 = kwargs.get('q2', 1.0)
    distance = kwargs.get('distance', 1.0)
    
    force_law, selected_law_version = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        true_force = force_law(q1, q2, distance)
        return inject_noise(true_force, noise_level, ABSOLUTE_FORCE_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**LINEAR_DEFAULTS, **kwargs}
        return _run_linear_coulomb_experiment(
            q1=q1,
            m1=kwargs.get('m1', 1.0),     # q1 mass (can be modified by user)
            q2=q2,
            m2=kwargs.get('m2', 1.0),     # q2 mass (can be modified by user)
            distance=distance,
            duration=params.get('duration', LINEAR_DEFAULTS['duration']),
            time_step=params.get('time_step', LINEAR_DEFAULTS['time_step']),
            noise_level=noise_level,
            force_law=force_law
        )
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**TWO_DIM_DEFAULTS, **kwargs}
        pos1 = kwargs.get('pos1', [0.0, 0.0])
        pos2 = kwargs.get('pos2', [distance, 0.0])
        return _run_linear_coulomb_experiment_with_kinetic_energy(
            q1=q1,
            m1=kwargs.get('m1', 1.0),     # q1 mass (can be modified by user)
            q2=q2,
            m2=kwargs.get('m2', 1.0),     # q2 mass (can be modified by user)
            distance=distance,
            duration=params.get('duration', LINEAR_DEFAULTS['duration']),
            time_step=params.get('time_step', LINEAR_DEFAULTS['time_step']),
            noise_level=noise_level,
            force_law=force_law
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
    Evaluator assessing the symbolic equivalence, and RMSLE of the LLM's submitted function.
    Args:
        llm_function_str: The submitted Python function as a string
        difficulty: Difficulty level ('easy', 'medium', 'hard')
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
    # np.random.seed(42)
    num_points = 5000
    # Use log-uniform sampling for all parameters
    q1_magnitudes = np.exp(np.random.uniform(np.log(1e-1), np.log(1e1), num_points))
    q2_magnitudes = np.exp(np.random.uniform(np.log(1e-1), np.log(1e1), num_points))
    
    test_data = {
        'q1': q1_magnitudes,
        'q2': q2_magnitudes,
        'distance': np.exp(np.random.uniform(np.log(1e-1), np.log(1e1), num_points)),
    }
    # Define parameter mapping for Coulomb force module
    parameter_mapping = {
        "q1": "q1",
        "q2": "q2", 
        "distance": "distance"
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
    