import numpy as np
from typing import Union, Dict, List, Any, Tuple, Optional, Callable
from utils.noise import inject_noise
import re

from .m2_types import (
    ExperimentSystem, 
    ABSOLUTE_POSITION_PRECISION, 
    ABSOLUTE_VELOCITY_PRECISION, 
    ABSOLUTE_FORCE_PRECISION,
    LINEAR_DEFAULTS,
    FIXED_WIRE_DEFAULTS,
)
from .physics import (
    calculate_acceleration_1d_magnetic,
    verlet_integration_1d
)
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

def validate_function_definition(code: str) -> Tuple[bool, str]:
    """Validate the LLM's function definition."""
    if not re.search(r'def\s+discovered_law\s*\(current1,\s*current2,\s*distance\):', code):
        return False, "Invalid function signature. Must be def discovered_law(current1, current2, distance):"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_fixed_wire_experiment(
    params: dict,
    noise_level: float,
    force_law: Callable
) -> dict:
    """
    Simulate the motion of a wire under the influence of AC/DC current interaction.
    
    Args:
        params: Dictionary containing the parameters for the experiment
        noise_level: Noise level for the experiment
        force_law: Callable function for calculating magnetic force
    """
    # Fixed parameters for this experiment
    frequency = 50.0  # Hz
    period = 1.0 / frequency  # 0.02 seconds
    num_points = 20  # Always return exactly 20 data points

    current1 = params['current1']
    current2 = params['current2']
    mass_wire = params['mass_wire']
    distance = params['distance']
    initial_velocity = params['initial_velocity']
    
    # Calculate time step to get exactly 20 points over 1 period
    time_step = period / (num_points - 1)
    
    # Generate time array
    times = np.linspace(0, period, num_points)
    
    # Initialize arrays for position and velocity
    positions = np.zeros(num_points)
    velocities = np.zeros(num_points)
    
    # Set initial conditions
    positions[0] = distance
    velocities[0] = initial_velocity
    
    # Simulate motion using Verlet integration
    for i in range(1, num_points):
        t = times[i-1]
        
        # Calculate instantaneous AC current for wire 1
        instantaneous_current1 = current1 * np.sin(2 * np.pi * frequency * t)
        
        # Calculate acceleration using the magnetic force law
        acc = calculate_acceleration_1d_magnetic(
            instantaneous_current1, current2, mass_wire, positions[i-1], force_law
        )
        
        # Use Verlet integration to update position and velocity
        pos_new, vel_half = verlet_integration_1d(
            positions[i-1], velocities[i-1], acc, time_step
        )
        
        # Calculate acceleration at new position for final velocity update
        acc_new = calculate_acceleration_1d_magnetic(
            instantaneous_current1, current2, mass_wire, pos_new, force_law
        )
        
        vel_new = vel_half + 0.5 * acc_new * time_step
        
        positions[i] = pos_new
        velocities[i] = vel_new
    
    # Inject noise into the measurements
    noisy_positions = inject_noise(positions, noise_level, ABSOLUTE_POSITION_PRECISION)
    noisy_velocities = inject_noise(velocities, noise_level, ABSOLUTE_VELOCITY_PRECISION)
    
    return {
        'time': ["{:.6e}".format(t) for t in times.tolist()],
        'position': ["{:.6e}".format(p) for p in noisy_positions.tolist()],
        'velocity': ["{:.6e}".format(v) for v in noisy_velocities.tolist()]
    }

def _run_linear_experiment(
    params: dict,
    noise_level: float,
    force_law: callable
) -> dict:
    """Simulate 1D linear motion of one current-carrying wire relative to another."""
    current1 = params['current1']
    current2 = params['current2']
    distance = params['distance']
    duration = params['duration']
    time_step = params['time_step']
    mass2 = params['mass_wire']
    initial_velocity = params['initial_velocity']
    
    num_steps = int(duration / time_step)
    if num_steps <= 0: return {'time': [], 'position': [], 'velocity': []}
    
    times, positions, velocities = np.arange(num_steps) * time_step, np.zeros(num_steps), np.zeros(num_steps)
    positions[0], velocities[0] = distance, initial_velocity
    
    for i in range(1, num_steps):
        acc = calculate_acceleration_1d_magnetic(current1, current2, mass2, positions[i-1], force_law)
        pos_new, vel_half = verlet_integration_1d(positions[i-1], velocities[i-1], acc, time_step)
        acc_new = calculate_acceleration_1d_magnetic(current1, current2, mass2, pos_new, force_law)
        vel_new = vel_half + 0.5 * acc_new * time_step
        positions[i], velocities[i] = pos_new, vel_new
    
    noisy_positions = inject_noise(positions, noise_level, ABSOLUTE_POSITION_PRECISION)
    noisy_velocities = inject_noise(velocities, noise_level, ABSOLUTE_VELOCITY_PRECISION)
    
    max_points = 20
    if len(times) > max_points:
        times, noisy_positions, noisy_velocities = times[:max_points], noisy_positions[:max_points], noisy_velocities[:max_points]
        
    return {
        'time': ["{:.3e}".format(t) for t in times.tolist()],
        'position': ["{:.6e}".format(p) for p in noisy_positions.tolist()],
        'velocity': ["{:.6e}".format(v) for v in noisy_velocities.tolist()]
    }

def run_experiment_for_module(
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: Optional[str] = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """Experiment runner for the Magnetism module."""
    force_law, _ = get_ground_truth_law(difficulty, law_version)

    if system == ExperimentSystem.VANILLA_EQUATION:
        current1 = kwargs['current1']
        current2 = kwargs['current2']
        distance = kwargs['distance']
        true_force = force_law(abs(current1), abs(current2), abs(distance))
        return inject_noise(true_force, noise_level, ABSOLUTE_FORCE_PRECISION)

    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**LINEAR_DEFAULTS, **kwargs}
        return _run_linear_experiment(
            params=params,
            noise_level=noise_level, 
            force_law=force_law
        )

    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**FIXED_WIRE_DEFAULTS, **kwargs}
        return _run_fixed_wire_experiment(
            params=params,
            noise_level=noise_level,
            force_law=force_law
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
    """Evaluator for the Magnetism module."""
    is_valid, validation_error = validate_function_definition(llm_function_str)
    if not is_valid:
        return {"rmsle": float('nan'), "exact_accuracy": 0.0, "symbolic_equivalent": False, "symbolic_msg": validation_error, "error": validation_error}

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    num_points = 5000
    test_data = {
        'current1': np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), num_points)),
        'current2': np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), num_points)),
        'distance': np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), num_points)),
    }

    parameter_mapping = {"current1": "current1", "current2": "current2", "distance": "distance"}
    
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