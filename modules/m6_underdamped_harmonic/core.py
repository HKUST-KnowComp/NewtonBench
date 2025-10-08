import numpy as np
from typing import Union, Dict, List, Any, Tuple, Optional, Callable
from utils.noise import inject_noise
import re

from .m6_types import (
    ExperimentSystem, 
    ABSOLUTE_ANGULAR_VELOCITY_PRECISION,
    ABSOLUTE_PERIOD_PRECISION,
    ABSOLUTE_AMPLITUDE_PRECISION,
    DAMPED_OSCILLATOR_DEFAULTS
)
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """Validate the LLM's function definition."""
    if not re.search(r'def\s+discovered_law\s*\(k,\s*m,\s*b\):', code):
        return False, "Invalid function signature. Must be def discovered_law(k, m, b):"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_damped_oscillator_simple(
    spring_constant: float,
    mass: float,
    damping_coefficient: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Calculates the period of a damped oscillator."""
    omega = ground_truth_law(spring_constant, mass, damping_coefficient)
    if not np.isfinite(omega) or omega == 0:
        return {'period': 'invalid'}
    
    period = 2 * np.pi / omega
    noisy_period = inject_noise(period, noise_level, ABSOLUTE_PERIOD_PRECISION)
    return {'period': "{:.6e}".format(float(noisy_period))}

def _run_damped_oscillator_difficult(
    spring_constant: float,
    mass: float,
    damping_coefficient: float,
    initial_amplitude: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Calculates the amplitude of a damped oscillator over time."""
    omega = ground_truth_law(spring_constant, mass, damping_coefficient)
    if not np.isfinite(omega) or omega == 0:
        return {'time': [], 'amplitude': []}

    period = 2 * np.pi / omega
    duration = 2 * period
    num_points = 20
    times = np.linspace(0, duration, num_points)
    
    amplitudes = initial_amplitude * np.exp(-3 * times) * np.cos(omega * times)
    noisy_amplitudes = inject_noise(amplitudes, noise_level, ABSOLUTE_AMPLITUDE_PRECISION)
    
    return {
        'time': ["{:.6e}".format(t) for t in times.tolist()],
        'amplitude': ["{:.6e}".format(a) for a in noisy_amplitudes.tolist()]
    }

def run_experiment_for_module(
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: Optional[str] = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """Experiment runner for the Underdamped Harmonic Motion module."""
    ground_truth_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        k = kwargs.get('k', DAMPED_OSCILLATOR_DEFAULTS['spring_constant'])
        m = kwargs.get('m', DAMPED_OSCILLATOR_DEFAULTS['mass'])
        b = kwargs.get('b', DAMPED_OSCILLATOR_DEFAULTS['damping_coefficient'])
        true_omega = ground_truth_law(k, m, b)
        return inject_noise(true_omega, noise_level, ABSOLUTE_ANGULAR_VELOCITY_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**DAMPED_OSCILLATOR_DEFAULTS, **kwargs}
        return _run_damped_oscillator_simple(
            spring_constant=params['spring_constant'],
            mass=params['mass'],
            damping_coefficient=params['damping_coefficient'],
            noise_level=noise_level,
            ground_truth_law=ground_truth_law
        )

    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**DAMPED_OSCILLATOR_DEFAULTS, **kwargs}
        return _run_damped_oscillator_difficult(
            spring_constant=params['spring_constant'],
            mass=params['mass'],
            damping_coefficient=params['damping_coefficient'],
            initial_amplitude=params['initial_amplitude'],
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
    """Evaluator for the Underdamped Harmonic Motion module."""
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    num_points = 5000
    test_data = {
        'k': np.exp(np.random.uniform(np.log(1e2), np.log(1e4), num_points)),
        'm': np.exp(np.random.uniform(np.log(1e-1), np.log(1e1), num_points)),
        'b': np.exp(np.random.uniform(np.log(1e-2), np.log(1e0), num_points))
    }

    parameter_mapping = {"k": "k", "m": "m", "b": "b"}
    
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
