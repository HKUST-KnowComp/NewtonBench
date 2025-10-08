import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Easy Difficulty Laws ---
def _ground_truth_law_easy_v0(k: float, x: float) -> float:
    """Easy Hooke's law: U = 2kx^2"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * k * (x ** 2)

def _ground_truth_law_easy_v1(k: float, x: float) -> float:
    """Easy Hooke's law: U = 2k^2x"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * k ** 2 * x

def _ground_truth_law_easy_v2(k: float, x: float) -> float:
    """Easy Hooke's law: U = 2k^3x^2"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * k ** 3 * x ** 2

# --- Medium Difficulty Laws ---
def _ground_truth_law_medium_v0(k: float, x: float) -> float:
    """Medium Hooke's law: U = 2(k + x^2)"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * (k + (x ** 2))

def _ground_truth_law_medium_v1(k: float, x: float) -> float:
    """Medium Hooke's law: U = 2(k^2 + x^3)"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * (k ** 2 + x ** 3)

def _ground_truth_law_medium_v2(k: float, x: float) -> float:
    """Medium Hooke's law: U = 2(k^3 + x^2)"""
    k, x = float(k), float(x)  # Ensure float conversion
    return 2 * (k ** 3 + (x ** 2))

# --- Hard Difficulty Laws ---
def _ground_truth_law_hard_v0(k: float, x: float) -> float:
    """Hard Hooke's law: U = 2(k + sin(x^2)), but clamped to 0 if negative"""
    k, x = float(k), float(x)  # Ensure float conversion for NumPy operations
    result = 2 * (k + np.sin(x ** 2))
    if result <= 0:
        return 0
    return 2 * (k + np.sin(x ** 2))

def _ground_truth_law_hard_v1(k: float, x: float) -> float:
    """Hard Hooke's law: U = 2(sin(k^2) + x^3), but clamped to 0 if negative"""
    k, x = float(k), float(x)  # Ensure float conversion for NumPy operations
    result = 2 * (np.sin(k ** 2) + x ** 3)
    if result <= 0:
        return 0
    return 2 * (np.sin(k ** 2) + x ** 3)

def _ground_truth_law_hard_v2(k: float, x: float) -> float:
    """Hard Hooke's law: U = 2(sin(k^3) + x^2), but clamped to 0 if negative"""
    k, x = float(k), float(x)  # Ensure float conversion for NumPy operations
    result = 2 * (np.sin(k ** 3) + (x ** 2))
    if result <= 0: 
        return 0
    return 2 * (np.sin(k ** 3) + (x ** 2))

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
