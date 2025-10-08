import numpy as np
from typing import Tuple, List, Dict, Any

def wrap_angle(angle):
    """
    Wraps any angle in degrees to the range [0, 90].

    Parameters:
    - angle (float): The input angle in degrees.

    Returns:
    - float: The wrapped angle in [0, 90].
    """
    angle = angle % 180  # First wrap to [0, 180)
    if angle > 90:
        angle = 180 - angle  # Reflect to [0, 90]
    return angle