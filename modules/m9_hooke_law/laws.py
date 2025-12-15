import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

CONSTANT = 231.141
CONSTANT2 = 1241.9012
CONSTANT3 = 12.578

# --- Easy Difficulty Laws ---
def _ground_truth_law_easy_v0(x: float) -> float:
    """Easy Hooke's law: U = 2kx^2"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * (x ** 2)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_easy_v1(x: float) -> float:
    """Easy Hooke's law: U = 2kx^0.5"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 0.5
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_easy_v2(x: float) -> float:
    """Easy Hooke's law: U = 2kx^3.4"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 3.4
    except (ValueError, ZeroDivisionError):
        return float('nan')

# --- Medium Difficulty Laws ---
def _ground_truth_law_medium_v0(x: float) -> float:
    """Medium Hooke's law: U = 2k x^2 + K2*x"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * (x ** 2) + CONSTANT2 * x
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v1(x: float) -> float:
    """Medium Hooke's law: U = 2k x^0.5 + K2*x^3"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 0.5 + CONSTANT2 * x ** 3
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v2(x: float) -> float:
    """Medium Hooke's law: U = 2k x^3.4 + K2*x^0.5"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 3.4 + CONSTANT2 * x ** 0.5
    except (ValueError, ZeroDivisionError):
        return float('nan')

# --- Hard Difficulty Laws ---
def _ground_truth_law_hard_v0(x: float) -> float:
    """Hard Hooke's law: U = 2k x^2 + K2*x + K3/sqrt(x)"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * (x ** 2) + CONSTANT2 * x + CONSTANT3 / (np.sqrt(x))
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v1(x: float) -> float:
    """Hard Hooke's law: U = 2k x^0.5 + K2*x^3 + K3/x^0.3"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 0.5 + CONSTANT2 * x ** 3 + CONSTANT3 / (x ** 0.3)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v2(x: float) -> float:
    """Hard Hooke's law: U = 2k x^3.4 + K2*x^0.5 + K3/x^(10/3)"""
    try:
        if x < 0:
            return float('nan')
        k, x = float(CONSTANT), float(x)  # Ensure float conversion
        return 2 * k * x ** 3.4 + CONSTANT2 * x ** 0.5 + CONSTANT3 / x ** (10/3)
    except (ValueError, ZeroDivisionError):
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
