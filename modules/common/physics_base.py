import numpy as np
from typing import Tuple

def verlet_integration_2d(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D Verlet integrator for orbital motion (shared by all modules).
    
    Args:
        pos: Current position [x, y]
        vel: Current velocity [vx, vy]
        acc: Current acceleration [ax, ay]
        dt: Time step
        
    Returns:
        Tuple of (new_position, new_velocity)
    """
    # Update position
    new_pos = pos + vel * dt + 0.5 * acc * dt**2
    
    # Store half-step velocity
    # Ensure two-stage update process that provides better numerical stability and energy conservation
    vel_half = vel + 0.5 * acc * dt
    
    return new_pos, vel_half

def verlet_integration_1d(
    pos: float,
    vel: float,
    acc: float,
    dt: float
) -> Tuple[float, float]:
    """
    1D Verlet integrator for linear motion (shared by all modules).
    
    Args:
        pos: Current position (scalar)
        vel: Current velocity (scalar)
        acc: Current acceleration (scalar)
        dt: Time step
        
    Returns:
        Tuple of (new_position, new_velocity)
    """
    # Update position
    new_pos = pos + vel * dt + 0.5 * acc * dt**2
    
    # Store half-step velocity
    # Ensure two-stage update process that provides better numerical stability and energy conservation
    vel_half = vel + 0.5 * acc * dt
    
    return new_pos, vel_half 