import numpy as np
from typing import Union, Dict, Any, Tuple, Optional, Callable
from utils.noise import inject_noise
import re
from scipy.integrate import quad

from .m10_types import (
    ExperimentSystem, 
    ABSOLUTE_SPECTRAL_RADIANCE_PRECISION,
    ABSOLUTE_OCCUPATION_NUMBER_PRECISION,
    ABSOLUTE_POWER_PRECISION,
    BLACK_BODY_DEFAULTS,
    DIFFICULT_MODEL_DEFAULTS
)
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """Validate the LLM's function definition."""
    if not re.search(r'def\s+discovered_law\s*\(\s*omega,\s*T\s*\):', code):
        return False, "Invalid function signature. Must be def discovered_law(omega, T):"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_black_body_spectrometer(
    temperature: float,
    probe_frequency: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Runs the black-body spectrometer experiment for a single frequency."""
    n = ground_truth_law(probe_frequency, temperature)
    radiance = n * (probe_frequency**3)
    noisy_radiance = inject_noise(radiance, noise_level, ABSOLUTE_SPECTRAL_RADIANCE_PRECISION)
    return {'spectral_radiance': f"{float(noisy_radiance):.12e}"}

def _run_difficult_model_experiment(
    temperature: float,
    center_frequency: float,
    bandwidth: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """Runs the photon gas calorimeter experiment."""
    
    def spectral_radiance(omega):
        n = ground_truth_law(omega, temperature)
        return n * (omega**3)

    lower_bound = center_frequency - bandwidth / 2
    upper_bound = center_frequency + bandwidth / 2
    
    total_power, _ = quad(spectral_radiance, lower_bound, upper_bound, limit=100)
    
    noisy_power = inject_noise(total_power, noise_level, ABSOLUTE_POWER_PRECISION)
    
    return {'total_power': f"{float(noisy_power):.6e}"}

def run_experiment_for_module(
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: Optional[str] = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """Experiment runner for the Bose-Einstein distribution module."""
    ground_truth_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        omega = kwargs.get('omega', 1e8)
        T = kwargs.get('temperature', 1e3)
        
        true_n = ground_truth_law(omega, T)
        return inject_noise(true_n, noise_level, ABSOLUTE_OCCUPATION_NUMBER_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**BLACK_BODY_DEFAULTS, **kwargs}
        return _run_black_body_spectrometer(
            temperature=params['temperature'],
            probe_frequency=params['probe_frequency'],
            noise_level=noise_level,
            ground_truth_law=ground_truth_law
        )

    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**DIFFICULT_MODEL_DEFAULTS, **kwargs}
        return _run_difficult_model_experiment(
            temperature=params['temperature'],
            center_frequency=params['center_frequency'],
            bandwidth=params['bandwidth'],
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
    """Evaluator for the Bose-Einstein distribution module."""
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    num_points = 5000
    test_data = {
        'omega': np.exp(np.random.uniform(np.log(1e8), np.log(1e10), num_points)),
        'T': np.exp(np.random.uniform(np.log(1e1), np.log(1e3), num_points)),
    }
                
    parameter_mapping = {
        "omega": "omega", 
        "T": "T",
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
