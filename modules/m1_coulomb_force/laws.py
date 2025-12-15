import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Environment Constants ---
CONSTANT = 2.0

# --- Ground Truth Physics Engine ---
def _ground_truth_law_easy_v0(q1: float, q2: float, distance: float) -> float:
    """Easy law: F = K * q1 * q2 / r^3"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * q1 * q2) / (distance ** 3)

def _ground_truth_law_easy_v1(q1: float, q2: float, distance: float) -> float:
    """Easy law: F = K * (q1 * q2)^3 / r^2"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * (q1 * q2) ** 3) / (distance ** 2)

def _ground_truth_law_easy_v2(q1: float, q2: float, distance: float) -> float:
    """Easy law: F = K * q1^3 * q2 / r^2"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * q1 ** 3 * q2) / (distance ** 2)

def _ground_truth_law_medium_v0(q1: float, q2: float, distance: float) -> float:
    """Medium law: F = K * (q1 * q2) * (q1 + q2) / r^2"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * (q1 * q2) * (q1 + q2)) / (distance ** 2)

def _ground_truth_law_medium_v1(q1: float, q2: float, distance: float) -> float:
    """Medium law: F = K * (q1 + q2)^3 / r^2"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * (q1 + q2) ** 3) / (distance ** 2)

def _ground_truth_law_medium_v2(q1: float, q2: float, distance: float) -> float:
    """Medium law: F = K * q1^3 * q2^2 / r^2.5"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * q1 ** 3 * q2 ** 2) / (distance ** 2.5)

def _ground_truth_law_hard_v0(q1: float, q2: float, distance: float) -> float:
    """Hard law: F = K * (q1 * q2) * (q1 + q2) / r^e"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return ((CONSTANT * (q1 * q2) * (q1 + q2)) / (distance ** np.exp(1)))

def _ground_truth_law_hard_v1(q1: float, q2: float, distance: float) -> float:
    """Hard law: F = K * q2^2 * (q1 + q2)^3 / r^2"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * q2 ** 2 * (q1 + q2) ** 3) / (distance ** 2)

def _ground_truth_law_hard_v2(q1: float, q2: float, distance: float) -> float:
    """Hard law: F = K * q1^3 * q2^2 / r^e"""
    if distance <= 0 or q1 == 0 or q2 == 0:
        return 0.0
    return (CONSTANT * q1 ** 3 * q2 ** 2) / (distance ** np.exp(1))

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
    },
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
