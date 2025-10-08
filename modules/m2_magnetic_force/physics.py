import numpy as np
from typing import Tuple, Callable, Dict, Any
from modules.common.physics_base import verlet_integration_2d, verlet_integration_1d

def calculate_acceleration_1d_magnetic(
    current1: float,
    current2: float,
    mass2: float,
    distance: float,
    force_law: Callable
) -> float:
    """
    Calculate 1D acceleration for wire2 due to the magnetic force created by wire1.
    """
    if mass2 == 0 or distance == 0:
        return 0.0
    
    force = force_law(abs(current1), abs(current2), abs(distance))
    if (np.sign(current1) == np.sign(current2)):
        # same direction attract
        acc = - force / mass2
    else:
        # opposite direction repel
        acc = force / mass2  
    return acc