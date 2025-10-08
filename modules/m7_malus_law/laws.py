
"""
Ground truth laws for Module 7: Malus's Law

This module contains the true mathematical relationships that govern
the transmission of polarized light through polarizers at different angles.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional


def _ground_truth_law_easy_v0(I_0: float, theta: float) -> float:
    """
    Easy Malus's Law: I = I_0 * (sin(theta) + cos(theta))^2
    
    This is the modified Malus's Law that describes how the intensity
    of polarized light changes as it passes through a polarizer.
    
    Args:
        I_0: Initial light intensity in W/m² (should be positive)
        theta: Angle between polarization direction and polarizer axis in radians
    
    Returns:
        Transmitted light intensity I in W/m² (returns 0.0 if I_0 is invalid)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.sin(theta) + np.cos(theta)) ** 2
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_easy_v1(I_0: float, theta: float) -> float:
    """
    Easy Malus's Law: I = I_0 * (sin(theta) / cos(theta))^2
    
    This is the modified Malus's Law that describes how the intensity
    of polarized light changes as it passes through a polarizer.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.sin(theta) / np.cos(theta)) ** 2
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_easy_v2(I_0: float, theta: float) -> float:
    """
    Easy Malus's Law: I = I_0 * (cos(theta) / sin(theta))^2
    
    This is the modified Malus's Law that describes how the intensity
    of polarized light changes as it passes through a polarizer.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.cos(theta) / np.sin(theta)) ** 2
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v0(I_0: float, theta: float) -> float:
    """
    Medium Malus's Law: I = I_0 * (2 * sin(theta) + cos(theta))^2
    
    This is a modified version of Malus's Law with a cubic power
    relationship, representing a more complex polarization system.
    
    Args:
        I_0: Initial light intensity in W/m² (should be positive)
        theta: Angle between polarization direction and polarizer axis in radians
    
    Returns:
        Transmitted light intensity I in W/m² (returns 0.0 if I_0 is invalid)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (2 * np.sin(theta) + np.cos(theta)) ** 2
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v1(I_0: float, theta: float) -> float:
    """
    Medium Malus's Law: I = I_0 * sin(theta)^2 / cos(theta)^3
    
    This is a modified version of Malus's Law with a cubic power
    relationship, representing a more complex polarization system.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * np.sin(theta) ** 2 / np.cos(theta) ** 3
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v2(I_0: float, theta: float) -> float:
    """
    Medium Malus's Law: I = I_0 * (cos(theta) / sin(theta))^e
    
    This is a modified version of Malus's Law with a cubic power
    relationship, representing a more complex polarization system.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.cos(theta) / np.sin(theta)) ** np.exp(1)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v0(I_0: float, theta: float) -> float:
    """
    Hard Malus's Law: I = I_0 * (2 * sin(theta) + 1.5 * cos(theta))^2
    
    This is an exponential decay relationship that represents
    a highly complex polarization system with quantum effects.
    
    Args:
        I_0: Initial light intensity in W/m² (should be positive)
        theta: Angle between polarization direction and polarizer axis in radians
    
    Returns:
        Transmitted light intensity I in W/m² (returns 0.0 if I_0 is invalid)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (2 * np.sin(theta) + 1.5 * np.cos(theta)) ** 2
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v1(I_0: float, theta: float) -> float:
    """
    Hard Malus's Law: I = I_0 * (sin(theta)^2 / cos(theta)^3)^e
    
    This is an exponential decay relationship that represents
    a highly complex polarization system with quantum effects.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.sin(theta) ** 2 / np.cos(theta) ** 3) ** np.exp(1)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v2(I_0: float, theta: float) -> float:
    """
    Hard Malus's Law: I = I_0 * (sin(theta)^2 / cos(theta))^e
    
    This is an exponential decay relationship that represents
    a highly complex polarization system with quantum effects.
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = I_0 * (np.sin(theta) ** 2 / np.cos(theta)) ** np.exp(1)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

# --- Law Registry ---
LAW_REGISTRY = {
    'easy': {
        'v0': _ground_truth_law_easy_v0,
        'v1': _ground_truth_law_easy_v1,
        'v2': _ground_truth_law_easy_v2,
    },
    'medium': {
        'v0': _ground_truth_law_medium_v0,
        'v1': _ground_truth_law_medium_v1,
        'v2': _ground_truth_law_medium_v2,
    },
    'hard': {
        'v0': _ground_truth_law_hard_v0,
        'v1': _ground_truth_law_hard_v1,
        'v2': _ground_truth_law_hard_v2,
    }
}

def get_ground_truth_law(difficulty: str, law_version: Optional[str] = None) -> Tuple[Callable, str]:
    """
    Get the ground truth law function for the specified difficulty and version.
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of {list(LAW_REGISTRY.keys())}")
    
    available_versions = list(LAW_REGISTRY[difficulty].keys())
    
    if law_version is None:
        law_version = random.choice(available_versions)
    elif law_version not in available_versions:
        raise ValueError(f"Law version '{law_version}' not found for difficulty '{difficulty}'. Available: {available_versions}")
    
    law_function = LAW_REGISTRY[difficulty][law_version]
    return law_function, law_version

def get_available_law_versions(difficulty: str) -> List[str]:
    """
    Get list of available law versions for a difficulty level.
    
    Args:
        difficulty: Difficulty level
        
    Returns:
        List of available version strings
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    return list(LAW_REGISTRY[difficulty].keys())
