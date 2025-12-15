import numpy as np
from typing import Union, Dict, Any, Tuple, Optional, Callable
from utils.noise import inject_noise
import re

from .m8_types import (
    ExperimentSystem,
    ABSOLUTE_VELOCITY_PRECISION,
    ABSOLUTE_TIME_PRECISION,
    ABSOLUTE_LENGTH_PRECISION,
    ECHO_METHOD_DEFAULTS,
    RESONANCE_TUBE_DEFAULTS
)
from .physics import calculate_echo_time, calculate_resonance_lengths
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """Validate the LLM's function definition."""
    if not re.search(r'def\s+discovered_law\s*\(\s*gamma,\s*T,\s*M\s*\):', code):
        return False, "Invalid function signature. Must be def discovered_law(gamma, T, M):"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_echo_experiment(
    adiabatic_index: float,
    molar_mass: float,
    temperature: float,
    distance: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Runs the echo method experiment."""
    speed_of_sound = ground_truth_law(adiabatic_index, temperature, molar_mass)
    
    if not np.isfinite(speed_of_sound) or speed_of_sound <= 0:
        return {'time': 'invalid'}

    echo_time = calculate_echo_time(distance, speed_of_sound)
    noisy_time = inject_noise(echo_time, noise_level, ABSOLUTE_TIME_PRECISION)
    
    return {'time': f"{float(noisy_time):.6e}"}

def _run_resonance_experiment(
    adiabatic_index: float,
    molar_mass: float,
    temperature: float,
    driving_frequency: float,
    tube_diameter: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Runs the resonance tube experiment."""
    speed_of_sound = ground_truth_law(adiabatic_index, temperature, molar_mass)

    if not np.isfinite(speed_of_sound) or speed_of_sound <= 0:
        return {'first_resonance_length': 'invalid', 'second_resonance_length': 'invalid'}

    l1, l2 = calculate_resonance_lengths(speed_of_sound, driving_frequency, tube_diameter)

    noisy_l1 = inject_noise(l1, noise_level, ABSOLUTE_LENGTH_PRECISION)
    noisy_l2 = inject_noise(l2, noise_level, ABSOLUTE_LENGTH_PRECISION)

    return {
        'first_resonance_length': f"{float(noisy_l1):.6e}",
        'second_resonance_length': f"{float(noisy_l2):.6e}"
    }

def run_experiment_for_module(
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: Optional[str] = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """Experiment runner for the Speed of Sound module."""
    ground_truth_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        gamma = kwargs.get('adiabatic_index', ECHO_METHOD_DEFAULTS['adiabatic_index'])
        T = kwargs.get('temperature', ECHO_METHOD_DEFAULTS['temperature'])
        M = kwargs.get('molar_mass', ECHO_METHOD_DEFAULTS['molar_mass'])
        
        true_speed = ground_truth_law(gamma, T, M)
        return inject_noise(true_speed, noise_level, ABSOLUTE_VELOCITY_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**ECHO_METHOD_DEFAULTS, **kwargs}
        return _run_echo_experiment(
            adiabatic_index=params['adiabatic_index'],
            molar_mass=params['molar_mass'],
            temperature=params['temperature'],
            distance=params['distance'],
            noise_level=noise_level,
            ground_truth_law=ground_truth_law
        )

    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**RESONANCE_TUBE_DEFAULTS, **kwargs}
        return _run_resonance_experiment(
            adiabatic_index=params['adiabatic_index'],
            molar_mass=params['molar_mass'],
            temperature=params['temperature'],
            driving_frequency=params['driving_frequency'],
            tube_diameter=params['tube_diameter'],
            noise_level=noise_level,
            ground_truth_law=ground_truth_law
        )
    else:
        raise ValueError(f"Invalid system: {system}")

def evaluate_law(
    llm_function_str: str,
    param_description: str,
    difficulty: str = 'easy',
    law_version: Optional[str] = None,
    judge_model_name: str = "nemotron-ultra",
    trial_info=None,
) -> dict:
    """Evaluator for the Speed of Sound module."""
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    num_points = 5000
    test_data = {
        'gamma': np.random.uniform(1.3, 1.7, num_points),
        'T': np.exp(np.random.uniform(np.log(1e1), np.log(1e3), num_points)),
        'M': np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), num_points))
    }
                
    parameter_mapping = {
        "gamma": "gamma", 
        "T": "T",
        "M": "M"
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
