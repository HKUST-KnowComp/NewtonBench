import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Environment Constants ---
HIDDEN_CONSTANT_C = 6.674e-5

# --- v0 laws ---
def _ground_truth_law_easy_v0(mass1: float, mass2: float, distance: float) -> float:
    """Easy law: F = C * m1 * m2 / r^1.5"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * mass1 * mass2) / (distance ** 1.5)

def _ground_truth_law_medium_v0(mass1: float, mass2: float, distance: float) -> float:
    """Medium law: F = C * (m1 * m2)^2 / r^1.5"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * (mass1 * mass2) ** 2) / (distance ** 1.5)

def _ground_truth_law_hard_v0(mass1: float, mass2: float, distance: float) -> float:
    """Hard law: F = C * (m1 + m2)^2 / r^1.5"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * (mass1 + mass2) ** 2) / (distance ** 1.5)

# --- v1 laws ---
def _ground_truth_law_easy_v1(mass1: float, mass2: float, distance: float) -> float:
    """Easy law: F = C * m1 / r^2"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * mass1) / (distance ** 2)

def _ground_truth_law_medium_v1(mass1: float, mass2: float, distance: float) -> float:
    """Medium law: F = C * m1 / r^2.6"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * mass1) / (distance ** 2.6)

def _ground_truth_law_hard_v1(mass1: float, mass2: float, distance: float) -> float:
    """Hard law: F = C * m1^1.3 / r^2.6"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * mass1 ** 1.3) / (distance ** 2.6)

# --- v2 laws ---
def _ground_truth_law_easy_v2(mass1: float, mass2: float, distance: float) -> float:
    """Easy law: F = C * (m1^2 * m2^2) / r^2"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * (mass1 ** 2 * mass2 ** 2)) / (distance ** 2)

def _ground_truth_law_medium_v2(mass1: float, mass2: float, distance: float) -> float:
    """Medium law: F = C * (m1^2 * m2^2) * r^2"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * (mass1 ** 2 * mass2 ** 2)) * (distance ** 2)

def _ground_truth_law_hard_v2(mass1: float, mass2: float, distance: float) -> float:
    """Hard law: F = C * (m1^2 + m2^2) * r^2"""
    if distance <= 0 or mass1 <= 0 or mass2 <= 0:
        return 0.0
    return (HIDDEN_CONSTANT_C * (mass1 ** 2 + mass2 ** 2)) * (distance ** 2)

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
    Get ground truth law function for the given difficulty and optional specific version.
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        law_version: Specific law version ('v0', 'v1', 'v2') or None for random selection
        
    Returns:
        Tuple of (law_function, selected_version)
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}. Choose from 'easy', 'medium', 'hard'.")
    
    available_versions = list(LAW_REGISTRY[difficulty].keys())
    
    if law_version is None:
        # Random selection for variety
        selected_version = random.choice(available_versions)
    else:
        if law_version not in available_versions:
            raise ValueError(f"Invalid law version '{law_version}' for difficulty '{difficulty}'. Available: {available_versions}")
        selected_version = law_version
    
    return LAW_REGISTRY[difficulty][selected_version], selected_version

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
