import math
import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Easy Difficulty Laws ---

def _ground_truth_law_easy_v0(k: float, A: float, delta_T: float, d: float) -> float:
    """Easy Fourier law: P = (k * A * delta_T) / d^2"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * A * delta_T) / (d ** 2)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_easy_v1(k: float, A: float, delta_T: float, d: float) -> float:
    """Easy Fourier law: P = (k * (A ** 0.5) * delta_T) / d"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * (A ** 0.5) * delta_T) / d
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_easy_v2(k: float, A: float, delta_T: float, d: float) -> float:
    """Easy Fourier law: P = (k * A * delta_T ** 2 / d)"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * A * delta_T ** 2) / d
    except (ValueError, ZeroDivisionError):
        return float('nan')

# --- Medium Difficulty Laws ---

def _ground_truth_law_medium_v0(k: float, A: float, delta_T: float, d: float) -> float:
    """Medium Fourier law: P = (k * (A ** 0.5) * delta_T) / d^2"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * (A ** 0.5) * delta_T) / (d ** 2)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v1(k: float, A: float, delta_T: float, d: float) -> float:
    """Medium Fourier law: P = (k * (A ** 0.5) * (delta_T ** 2.7)) / d"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * (A ** 0.5) * delta_T ** 2.7) / d
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v2(k: float, A: float, delta_T: float, d: float) -> float:
    """Medium Fourier law: P = (k * delta_T ** 2) / (d * (A ** 3.4))"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * delta_T ** 2) / (d * (A ** 3.4))
    except (ValueError, ZeroDivisionError):
        return float('nan')

# --- Hard Difficulty Laws ---

def _ground_truth_law_hard_v0(k: float, A: float, delta_T: float, d: float) -> float:
    """Hard Fourier law: P = (k * (A ** 0.5) * (delta_T ** 1.3)) / d^2"""
    try:
        if delta_T <= 0 or d <= 0:
            return 0.0
        return (k * (A ** 0.5) * (delta_T**1.3)) / (d ** 2)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v1(k: float, A: float, delta_T: float, d: float) -> float:
    """Hard Fourier law: P = (k * (A ** 0.5) * (delta_T ** 2.7)) / d ** (3/7)"""
    try:
        if d <= 0 or delta_T <= 0:
            return 0.0
        return (k * (A ** 0.5) * delta_T ** 2.7) / d ** (3/7)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v2(k: float, A: float, delta_T: float, d: float) -> float:
    """Hard Fourier law: P = (k * delta_T ** 2) / (sqrt(d) * (A ** 3.4))"""
    try:
        if delta_T <= 0 or d <= 0:
            return 0.0
        return (k * delta_T ** 2) / (np.sqrt(d) * (A ** 3.4))
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
