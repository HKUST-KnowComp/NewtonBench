# utils/noise.py

import random
import numpy as np

def inject_noise(true_result, noise_level: float, absolute_noise_floor: float = 1e-9):
    """
    Applies a level-tunable, relative Gaussian noise to a true experimental result.
    Supports both scalar floats and NumPy arrays.

    Args:
        true_result (float or np.ndarray): The perfect, noise-free result from the physics engine.
        noise_level (float): The relative standard deviation for the noise (e.g., 0.01 for 1%).
                             If 0, the function returns the true_result exactly.
        absolute_noise_floor (float): The minimum possible standard deviation for the noise,
                                      simulating instrument precision limits.

    Returns:
        float or np.ndarray: The result with added Gaussian noise.
    """
    if noise_level == 0.0:
        return true_result

    if isinstance(true_result, np.ndarray):
        sigma = np.maximum(np.abs(true_result * noise_level), absolute_noise_floor)
        noise = np.random.normal(loc=0, scale=sigma, size=true_result.shape)
        return true_result + noise
    else:
        sigma = max(abs(true_result * noise_level), absolute_noise_floor)
        # Generate noise from a normal (Gaussian) distribution and add it
        noise = random.gauss(mu=0, sigma=sigma)
        return true_result + noise