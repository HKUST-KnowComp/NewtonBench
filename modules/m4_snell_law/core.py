import numpy as np
from typing import Union, Dict, List, Any, Tuple, Optional, Callable
from utils.noise import inject_noise
import re

from .m4_types import (
    ExperimentSystem, 
    ABSOLUTE_ANGLE_PRECISION, 
    LIGHT_PROPAGATION_DEFAULTS,
    TRIPLE_LAYER_DEFAULTS,
    SPEED_OF_LIGHT
)
from .physics import (
    wrap_angle
)
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """Validate the LLM's function definition."""
    if not re.search(r'def\s+discovered_law\s*\(n1,\s*n2,\s*angle1\):', code):
        return False, "Invalid function signature. Must be def discovered_law(n1, n2, angle1):"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_light_propagation_experiment(
    speed_medium1: float,
    speed_medium2: float,
    incidence_angle: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """
    Two-medium measurement for the simple system.
    Calculates refractive indices from speeds, then uses the ground truth law
    to find the refraction angle.
    Returns a single measurement object with the formatted refraction angle.
    """
    # Calculate refractive indices from speeds
    n1 = SPEED_OF_LIGHT / speed_medium1
    n2 = SPEED_OF_LIGHT / speed_medium2

    # Compute angle2 from the selected ground truth law using (n1, n2, angle1)
    # Ensure incidence angle is wrapped to valid range
    theta1_deg = wrap_angle(incidence_angle)
    theta2_deg = float(ground_truth_law(n1, n2, theta1_deg))
    
    # Guard against non-physical or invalid angles
    if not np.isfinite(theta2_deg):
        theta2_deg = None

    if theta2_deg is not None:
        # Inject measurement noise
        noisy_theta2 = inject_noise(theta2_deg, noise_level, ABSOLUTE_ANGLE_PRECISION)

        return {
            'refraction_angle': "{:.6e}".format(float(noisy_theta2)),
        }
    else:
        return {
            'refraction_angle': "invalid",
        }

def _run_triple_layer_experiment(
    refractive_index_1: float,
    refractive_index_2: float,
    refractive_index_3: float,
    incidence_angle: float,
    noise_level: float,
    ground_truth_law: Callable
) -> dict:
    """
    Simulate light passing through three media layers.

    This function calculates the final refraction angle by applying the ground
    truth law (Snell's Law) sequentially at each interface.
    """
    # Calculate refraction angle from medium 1 to medium 2
    theta2_deg = float(ground_truth_law(refractive_index_1, refractive_index_2, incidence_angle))

    # Guard against non-physical or invalid intermediate angles
    if not np.isfinite(theta2_deg):
        return {'final_refraction_angle': "invalid"}

    # Use the angle from medium 2 as the incidence angle for medium 3
    # The angle of incidence for the second interface is the same as the angle of refraction from the first
    theta3_deg = float(ground_truth_law(refractive_index_2, refractive_index_3, theta2_deg))

    # Guard against non-physical or invalid final angles
    if not np.isfinite(theta3_deg):
        return {'final_refraction_angle': "invalid"}

    # Inject measurement noise into the final angle
    noisy_theta3 = inject_noise(theta3_deg, noise_level, ABSOLUTE_ANGLE_PRECISION)

    return {
        'final_refraction_angle': "{:.6e}".format(float(noisy_theta3))
    }

def run_experiment_for_module(
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: Optional[str] = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """Experiment runner for the Snell's Law module."""
    ground_truth_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        refractive_index_1 = kwargs.get('refractive_index_1', 1.1)
        refractive_index_2 = kwargs.get('refractive_index_2', 1.2)
        incidence_angle = kwargs.get('incidence_angle', 0.0)

        incidence_angle = wrap_angle(incidence_angle)
        
        # Calculate the refraction angle
        true_refraction_angle = ground_truth_law(refractive_index_1, refractive_index_2, incidence_angle)
        
        # Inject noise into the measurement
        return inject_noise(true_refraction_angle, noise_level, ABSOLUTE_ANGLE_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        # Extract parameters for adjusted two-medium measurement experiment
        speed_medium1 = kwargs.get('speed_medium1', LIGHT_PROPAGATION_DEFAULTS['speed_medium1'])
        speed_medium2 = kwargs.get('speed_medium2', LIGHT_PROPAGATION_DEFAULTS['speed_medium2'])
        incidence_angle = kwargs.get('incidence_angle', LIGHT_PROPAGATION_DEFAULTS['incidence_angle'])
        incidence_angle = wrap_angle(incidence_angle)

        return _run_light_propagation_experiment(
            speed_medium1=speed_medium1,
            speed_medium2=speed_medium2,
            incidence_angle=incidence_angle,
            noise_level=noise_level,
            ground_truth_law=ground_truth_law
        )

    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        # Extract parameters for triple-layer experiment
        refractive_index_1 = kwargs.get('refractive_index_1', TRIPLE_LAYER_DEFAULTS['refractive_index_1'])
        refractive_index_2 = kwargs.get('refractive_index_2', TRIPLE_LAYER_DEFAULTS['refractive_index_2'])
        refractive_index_3 = kwargs.get('refractive_index_3', TRIPLE_LAYER_DEFAULTS['refractive_index_3'])
        incidence_angle = kwargs.get('incidence_angle', TRIPLE_LAYER_DEFAULTS['incidence_angle'])
        incidence_angle = wrap_angle(incidence_angle)
                
        return _run_triple_layer_experiment(
            refractive_index_1=refractive_index_1,
            refractive_index_2=refractive_index_2,
            refractive_index_3=refractive_index_3,
            incidence_angle=incidence_angle,
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
    """Evaluator for the Snell's Law module."""
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    
    # Generate test data covering a wide range of scenarios
    num_points = 5000
    test_data = {
        'n1': np.random.uniform(1.0, 1.5, num_points),
        'n2': np.random.uniform(1.0, 1.5, num_points),
        'angle1': np.random.uniform(0.0, 90.0, num_points), 
    }
                
    parameter_mapping = {
        "n1": "n1", 
        "n2": "n2",
        "angle1": "angle1"
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