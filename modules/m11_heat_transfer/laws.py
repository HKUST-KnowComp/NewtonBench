import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Easy Difficulty Laws (v0 only) ---
def _ground_truth_law_easy_v0(m: float, c: float, delta_T: float) -> float:
    """
    Easy heat transfer law: Q = m * c * (delta_T)^2.5
    
    Note: This law assumes delta_T > 0 (temperature increase only).
    For cooling processes, use the absolute value of temperature change.
    
    Args:
        m: Mass in kg (should be positive)
        c: Specific heat capacity in J/(kg·K) (should be positive)
        delta_T: Temperature change in K (should be positive, represents heating)
    
    Returns:
        Heat transfer Q in Joules (returns 0.0 if any parameter is invalid)
    """
    # Check parameter validity and output 0 for invalid parameters
    if m <= 0 or c <= 0 or delta_T <= 0:
        return 0.0
    
    return m * c * (delta_T ** 2.5)

def _ground_truth_law_easy_v1(m: float, c: float, delta_T: float) -> float:
    """
    Easy heat transfer law: Q = m^2.5 * c * (delta_T)
    """
    return m ** 2.5 * c * (delta_T)

def _ground_truth_law_easy_v2(m: float, c: float, delta_T: float) -> float:
    """
    Easy heat transfer law: Q = m * c^2.5 * (delta_T)
    """
    return m * c ** 2.5 * (delta_T)

# --- Medium Difficulty Laws (v0 only) ---
def _ground_truth_law_medium_v0(m: float, c: float, delta_T: float) -> float:
    """
    Medium heat transfer law: Q = m * (delta_T)^2.5 * exp(-c)
    
    Note: This law assumes delta_T > 0 (temperature increase only).
    For cooling processes, use the absolute value of temperature change.
    
    Args:
        m: Mass in kg (should be positive)
        c: Specific heat capacity in J/(kg·K) (should be positive)
        delta_T: Temperature change in K (should be positive, represents heating)
    
    Returns:
        Heat transfer Q in Joules (returns 0.0 if any parameter is invalid)
    """
    # Check parameter validity and output 0 for invalid parameters
    if m <= 0 or c <= 0 or delta_T <= 0:
        return 0.0
    
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = m * (delta_T ** 2.5) * np.exp(-c)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v1(m: float, c: float, delta_T: float) -> float:
    """
    Medium heat transfer law: Q = m^2.5 * c * exp(delta_T^2)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = m ** 2.5 * c * np.exp(delta_T ** 2)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v2(m: float, c: float, delta_T: float) -> float:
    """
    Medium heat transfer law: Q = c^2.5 * exp(-delta_T * m)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = c ** 2.5 * np.exp(-delta_T * m)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

# --- Hard Difficulty Laws (v0 only) ---
def _ground_truth_law_hard_v0(m: float, c: float, delta_T: float) -> float:
    """
    Hard heat transfer law: Q = log(m * (delta_T ** 2.5)) * exp(-c)
    
    Note: This law assumes delta_T > 0 (temperature increase only).
    For cooling processes, use the absolute value of temperature change.
    
    Args:
        m: Mass in kg (should be positive)
        c: Specific heat capacity in J/(kg·K) (should be positive)
        delta_T: Temperature change in K (should be positive, represents heating)
    
    Returns:
        Heat transfer Q in Joules (returns 0.0 if any parameter is invalid, otherwise can be positive or negative due to sine function)
    """
    # Check parameter validity and output 0 for invalid parameters
    if m <= 0 or c <= 0 or delta_T <= 0:
        return 0.0
    
    # return np.sin(m * (delta_T ** 2.5)) * np.exp(-c)
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = np.log(m * (delta_T ** 2.5)) * np.exp(-c)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v1(m: float, c: float, delta_T: float) -> float:
    """
    Hard heat transfer law: Q = log(m^2.5 * c) * exp(-delta_T^2)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            # value = np.log(m * (delta_T ** 2.5)) * np.exp(delta_T ** 2)
            value = np.log(m ** 2.5 * c) * np.exp(-delta_T ** 2)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v2(m: float, c: float, delta_T: float) -> float:
    """
    Hard heat transfer law: Q = log(c * m^2.5) * exp(-delta_T * m)
    """
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = np.log(c ** 2.5) * np.exp(-delta_T + m)
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
