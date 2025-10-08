from typing import Callable, Tuple, Dict, List, Optional
import math
import random

# --- Ground Truth Laws ---

# --- v0 laws ---
def _ground_truth_law_easy_v0(k: float, m: float, b: float) -> float:
    """Easy law: ω = sqrt(k/m - (b/(2*m)))"""
    try:
        return math.sqrt(k/m - (b/(2*m)))
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v0(k: float, m: float, b: float) -> float:
    """Medium law: ω = sqrt(k/m - (b/(2*m**2)))"""
    try:
        return math.sqrt(k/m - (b/(2*m**2)))
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v0(k: float, m: float, b: float) -> float:
    """Hard law: ω = (k/m - (b/(2*m**2)))**1.5"""
    try:
        base = k/m - (b/(2*m**2))
        if base < 0:
            return float('nan')
        return (k/m - (b/(2*m**2)))**1.5
    except ValueError:
        return float('nan')
    
# --- v1 laws ---
def _ground_truth_law_easy_v1(k: float, m: float, b: float) -> float:
    """Easy law: ω = (k/m - (b/(2*m)) ^ 2) ^ 2"""
    try:
        return ((k/m) - (b/(2*m)) ** 2) ** 2
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v1(k: float, m: float, b: float) -> float:
    """Medium law: ω = (k/(m ^ 2) - (b/(2*m)) ^ 2) ^ 2"""
    try:
        return ((k/(m ** 2)) - (b/(2*m)) ** 2) ** 2
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v1(k: float, m: float, b: float) -> float:
    """Hard law: ω = (k*(m ^ 2) - (b/(2*m)) ^ 2) ^ 2"""
    try:
        return ((k*(m ** 2)) - (b/(2*m)) ** 2) ** 2
    except ValueError:
        return float('nan')
    
# --- v2 laws ---
def _ground_truth_law_easy_v2(k: float, m: float, b: float) -> float:
    """Easy law: ω = k/m - (b/(2*m)) ^ 2"""
    try:
        return (k/m - (b/(2*m)) ** 2)
    except ValueError:
        return float('nan')

def _ground_truth_law_medium_v2(k: float, m: float, b: float) -> float:
    """Medium law: ω = k/(m ^ 1.3) - (b/(2*m)) ^ 2"""
    try:
        return (k/(m ** 1.3) - (b/(2*m)) ** 2)
    except ValueError:
        return float('nan')

def _ground_truth_law_hard_v2(k: float, m: float, b: float) -> float:
    """Hard law: ω = k/(m ^ 1.3) - (b/(2*m)) ^ 0.7"""
    try:
        return (k/(m ** 1.3) - (b/(2*m)) ** 0.7)
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
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    return list(LAW_REGISTRY[difficulty].keys())
