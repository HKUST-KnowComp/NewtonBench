import numpy as np
from typing import Union, Dict, List, Any, Tuple
from utils.noise import inject_noise
import re
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .m0_types import (
    ExperimentSystem,
    ABSOLUTE_FORCE_PRECISION,
    ABSOLUTE_POSITION_PRECISION,
    ABSOLUTE_VELOCITY_PRECISION,
    TWO_DIM_DEFAULTS,
    LINEAR_DEFAULTS
)
from .physics import (
    verlet_integration_2d,
    verlet_integration_1d,
    calculate_acceleration_2d,
    calculate_acceleration_1d,
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
    if not re.search(r'def\s+discovered_law\s*\(mass1,\s*mass2,\s*distance\):', code):
        return False, "Invalid function signature"
    # Check if function has a return statement
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None

def _run_orbital_experiment(
    mass1: float,
    mass2: float,
    distance: float,
    initial_velocity: float,
    duration: float,
    time_step: float,
    noise_level: float,
    force_law: callable
) -> dict:
    """
    Simulate a 2D orbital motion experiment for the complex_system.
    Args:
        mass1 (float): Mass of the fixed central object.
        mass2 (float): Mass of the orbiting object.
        distance (float): Starting distance from origin.
        initial_velocity (float): Initial velocity magnitude (perpendicular to radius).
        duration (float): Time to track motion.
        time_step (float): Time interval between measurements.
        noise_level (float): Relative noise level for measurements.
        force_law (callable): Function to compute the force law.
    Returns:
        dict: Time series data with keys 'time', 'position', 'velocity', all as JSON-serializable lists (no NumPy arrays).
    """
    # Initialize arrays for time series data
    num_steps = int(duration / time_step)
    
    # Validate parameters to prevent empty arrays
    if num_steps <= 0:
        return {
            'time': [],
            'position': [],
            'velocity': []
        }
    
    times = np.arange(num_steps) * time_step
    positions = np.zeros((num_steps, 2))
    velocities = np.zeros((num_steps, 2))
    
    # Set initial conditions
    # mass1 at origin, mass2 at (distance, 0)
    positions[0] = np.array([distance, 0.0])
    # Initial velocity perpendicular to radius (for circular orbit)
    velocities[0] = np.array([0.0, initial_velocity])
    
    # Time evolution using Verlet integration
    for i in range(1, num_steps):
        # Calculate acceleration
        acc = calculate_acceleration_2d(
            mass1, mass2,
            np.array([0.0, 0.0]),  # mass1 at origin
            positions[i-1],
            force_law
        )[1]  # We only need acc of mass2
        
        # Update position and velocity
        pos_new, vel_half = verlet_integration_2d(
            positions[i-1],
            velocities[i-1],
            acc,
            time_step
        )
        
        # Calculate new acceleration for velocity update
        acc_new = calculate_acceleration_2d(
            mass1, mass2,
            np.array([0.0, 0.0]),
            pos_new,
            force_law
        )[1]
        
        # Final velocity update
        vel_new = vel_half + 0.5 * acc_new * time_step
        
        # Store results
        positions[i] = pos_new
        velocities[i] = vel_new
    
    # Add noise to measurements
    noisy_positions = inject_noise(positions, noise_level, ABSOLUTE_POSITION_PRECISION)
    noisy_velocities = inject_noise(velocities, noise_level, ABSOLUTE_VELOCITY_PRECISION)

    # Downsample to at most 20 data points
    max_points = 20
    if len(times) > max_points:
        times = times[:max_points]
        noisy_positions = noisy_positions[:max_points]
        noisy_velocities = noisy_velocities[:max_points]

    time_list = ["{:.3e}".format(float(t)) for t in times.tolist()]
    pos_list = [["{:.6e}".format(float(x)) for x in p] for p in noisy_positions.tolist()]
    vel_list = [["{:.6e}".format(float(x)) for x in v] for v in noisy_velocities.tolist()]
    return {
        'time': time_list,
        'position': pos_list,
        'velocity': vel_list
    }

def _run_linear_experiment(
    mass1: float,
    mass2: float,
    distance: float,
    initial_velocity: float,
    duration: float,
    time_step: float,
    noise_level: float,
    force_law: callable
) -> dict:
    """
    Simulate a 1D linear motion experiment for the simple_system.
    Args:
        mass1 (float): Mass of the fixed object at origin.
        mass2 (float): Mass of the moving object.
        distance (float): Starting position on x-axis.
        initial_velocity (float): Initial velocity (positive or negative).
        duration (float): Time to track motion.
        time_step (float): Time interval between measurements.
        noise_level (float): Relative noise level for measurements.
        force_law (callable): Function to compute the force law.
    Returns:
        dict: Time series data with keys 'time', 'position', 'velocity', all as JSON-serializable lists (no NumPy arrays).
    """
    # Initialize arrays for time series data
    num_steps = int(duration / time_step)
    
    # Validate parameters to prevent empty arrays
    if num_steps <= 0:
        return {
            'time': [],
            'position': [],
            'velocity': []
        }
    
    times = np.arange(num_steps) * time_step
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    accelerations = np.zeros(num_steps)
    
    # Set initial conditions
    positions[0] = distance
    velocities[0] = initial_velocity
    
    # Initial acceleration
    acc0 = calculate_acceleration_1d(
        mass1, mass2,
        positions[0],
        force_law
    )[1]  # We only need acc of mass2
    accelerations[0] = acc0
    
    # Time evolution using Verlet integration
    for i in range(1, num_steps):
        # Update position and velocity
        pos_new, vel_half = verlet_integration_1d(
            positions[i-1],
            velocities[i-1],
            accelerations[i-1],
            time_step
        )
        
        # Calculate new acceleration for velocity update
        acc_new = calculate_acceleration_1d(
            mass1, mass2,
            pos_new,
            force_law
        )[1]
        
        # Final velocity update
        vel_new = vel_half + 0.5 * acc_new * time_step
        
        # Store results
        positions[i] = pos_new
        velocities[i] = vel_new
        accelerations[i] = acc_new
    
    # Add noise to measurements
    noisy_positions = inject_noise(positions, noise_level, ABSOLUTE_POSITION_PRECISION)
    noisy_velocities = inject_noise(velocities, noise_level, ABSOLUTE_VELOCITY_PRECISION)

    # Downsample to at most 20 data points
    max_points = 20
    if len(times) > max_points:
        times = times[:max_points]
        noisy_positions = noisy_positions[:max_points]
        noisy_velocities = noisy_velocities[:max_points]

    time_list = ["{:.3e}".format(float(t)) for t in times.tolist()]
    pos_list = ["{:.6e}".format(float(x)) for x in noisy_positions.tolist()]
    vel_list = ["{:.6e}".format(float(x)) for x in noisy_velocities.tolist()]

    return {
        'time': time_list,
        'position': pos_list,
        'velocity': vel_list
    }

def run_experiment_for_module(
    mass1: float,
    mass2: float,
    distance: float,
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Enhanced experiment runner supporting vanilla_equation, simple_system (linear), and complex_system (orbital) modes.
    
    Args:
        mass1: Mass of first object
        mass2: Mass of second object
        distance: Initial distance between objects (used for all modes)
        noise_level: Relative noise level for measurements
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        **kwargs: Additional parameters
            For simple_system (linear motion):
                - initial_velocity: float
                - duration: float
                - time_step: float
            For complex_system (orbital motion):
                - initial_velocity: float
                - duration: float
                - time_step: float
    Returns:
        For vanilla_equation: force measurement (float)
        For simple/complex_system: time series data (dict)
    """
    # Get the appropriate force law
    force_law, selected_law_version = get_ground_truth_law(difficulty, law_version)

    # Vanilla equation (direct force measurement)
    if system == ExperimentSystem.VANILLA_EQUATION:
        true_force = force_law(mass1, mass2, distance)
        return inject_noise(true_force, noise_level, ABSOLUTE_FORCE_PRECISION)

    # Simple system (linear motion)
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        params = {**LINEAR_DEFAULTS, **kwargs}
        return _run_linear_experiment(
            mass1=mass1,
            mass2=mass2,
            distance=distance,
            initial_velocity=params.get('initial_velocity', 0.0),
            duration=params.get('duration', LINEAR_DEFAULTS['duration']),
            time_step=params.get('time_step', LINEAR_DEFAULTS['time_step']),
            noise_level=noise_level,
            force_law=force_law
        )

    # Complex system (orbital motion)
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        params = {**TWO_DIM_DEFAULTS, **kwargs}
        return _run_orbital_experiment(
            mass1=mass1,
            mass2=mass2,
            distance=distance,
            initial_velocity=params.get('initial_velocity', 0.0),
            duration=params.get('duration', TWO_DIM_DEFAULTS['duration']),
            time_step=params.get('time_step', TWO_DIM_DEFAULTS['time_step']),
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
    """Evaluator for the Gravity module."""
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
    num_points = 5000
    # Use log-uniform sampling for all parameters
    test_data = {
        'mass1': np.exp(np.random.uniform(np.log(1), np.log(1e3), num_points)),
        'mass2': np.exp(np.random.uniform(np.log(1), np.log(1e3), num_points)),
        'distance': np.exp(np.random.uniform(np.log(1), np.log(1e1), num_points)),
    }
    # Define parameter mapping for gravity module
    parameter_mapping = {
        "mass1": "mass1",
        "mass2": "mass2", 
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