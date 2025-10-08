import numpy as np
from typing import Tuple, Callable
from modules.common.physics_base import verlet_integration_2d, verlet_integration_1d

def calculate_acceleration_2d(
    mass1: float,
    mass2: float,
    pos1: np.ndarray,
    pos2: np.ndarray,
    force_law: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 2D acceleration vectors from force law.
    
    Args:
        mass1: Mass of first object
        mass2: Mass of second object
        pos1: Position vector of first object [x1, y1]
        pos2: Position vector of second object [x2, y2]
        force_law: Function that computes force magnitude
        
    Returns:
        Tuple of acceleration vectors (acc1, acc2)
    """
    # Calculate displacement vector
    r = pos2 - pos1
    distance = np.linalg.norm(r)

    if distance == 0:
        return np.zeros(2), np.zeros(2)
    
    # Unit vector pointing from pos1 to pos2
    r_hat = r / distance
    
    # Calculate force magnitude
    force = force_law(mass1, mass2, distance)
    
    # Calculate accelerations (Newton's second law)
    acc1 = (force / mass1) * r_hat
    acc2 = -(force / mass2) * r_hat
    
    return acc1, acc2

def calculate_acceleration_1d(
    mass1: float,
    mass2: float,
    distance: float,
    force_law: Callable
) -> Tuple[float, float]:
    """
    Calculate 1D acceleration from force law.
    
    Args:
        mass1: Mass of first (fixed) object
        mass2: Mass of second (moving) object
        distance: Scalar distance between objects
        force_law: Function that computes force magnitude
        
    Returns:
        Tuple of accelerations (acc1, acc2)
    """
    # Calculate force magnitude
    force = force_law(mass1, mass2, abs(distance))
    
    # Force direction depends on sign of distance
    force *= np.sign(distance)
    
    # Fixed mass has no acceleration
    acc1 = 0.0
    # Moving mass accelerates according to F = ma
    acc2 = -force / mass2
    
    return acc1, acc2