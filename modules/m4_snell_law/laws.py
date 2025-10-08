from typing import Callable, Tuple, Dict, List, Optional
import math
import random

# --- Ground Truth Laws ---

# --- v0 laws ---
def _ground_truth_law_easy_v0(n1: float, n2: float, angle1: float) -> float:
    """Easy law: angle2 = acos(n1 * sin(angle1) / n2)"""
    try:
        return math.degrees(math.acos(n1 * math.sin(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v0(n1: float, n2: float, angle1: float) -> float:
    """Medium law: angle2 = acos(n1 * cos(angle1) / n2)"""
    try:
        return math.degrees(math.acos(n1 * math.cos(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v0(n1: float, n2: float, angle1: float) -> float:
    """Hard law: angle2 = acos((n1 ^ 2) * cos(angle1) / n2)"""
    try:
        return math.degrees(math.acos((n1 ** 2) * math.cos(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')
    
# --- v1 laws ---
def _ground_truth_law_easy_v1(n1: float, n2: float, angle1: float) -> float:
    """Easy law: angle2 = asin(n2 * sin(angle1) / n1)"""
    try:
        return math.degrees(math.asin(n2 * math.sin(math.radians(angle1)) / n1))
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v1(n1: float, n2: float, angle1: float) -> float:
    """Medium law: angle2 = acos(n2 * sin(angle1) / n1)"""
    try:
        return math.degrees(math.acos(n2 * math.sin(math.radians(angle1)) / n1))
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v1(n1: float, n2: float, angle1: float) -> float:
    """Hard law: angle2 = acos(n2 * sin(angle1) / n1 ^ 2.5)"""
    try:
        return math.degrees(math.acos(n2 * math.sin(math.radians(angle1)) / (n1 ** 2.5)))
    except ValueError:
        return float('nan')
    
# --- v2 laws ---
def _ground_truth_law_easy_v2(n1: float, n2: float, angle1: float) -> float:
    """Easy law: angle2 = atan(n1 * sin(angle1) / n2)"""
    try:
        return math.degrees(math.atan(n1 * math.sin(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v2(n1: float, n2: float, angle1: float) -> float:
    """Medium law: angle2 = atan(n1 * tan(angle1) / n2)"""
    try:
        return math.degrees(math.atan(n1 * math.tan(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v2(n1: float, n2: float, angle1: float) -> float:
    """Hard law: angle2 = atan((n1/n2) ^ 2 * tan(angle1))"""
    try:
        return math.degrees(math.atan((n1 / n2) ** 2 * math.tan(math.radians(angle1))))
    except ValueError:
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
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        law_version: Specific law version (e.g., 'v0') or None for random selection
    
    Returns:
        Tuple of (law_function, law_version_string)
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
        List of available law version strings
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    return list(LAW_REGISTRY[difficulty].keys())