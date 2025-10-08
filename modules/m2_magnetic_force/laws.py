import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Environment Constants ---
MAGNETIC_CONSTANT_K = 4.78e-2

# --- v0 laws ---
def _ground_truth_law_easy_v0(current1: float, current2: float, distance: float) -> float:
    """Easy law: F = (K * I1 * I2) / (r ^ 3)"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * current1 * current2) / (distance ** 3)

def _ground_truth_law_medium_v0(current1: float, current2: float, distance: float) -> float:
    """Medium law: F = (K * (I1 * I2) ^ 1.5) / (r ^ 3)"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current1 * current2) ** (1.5)) / (distance ** 3)

def _ground_truth_law_hard_v0(current1: float, current2: float, distance: float) -> float:
    """Hard law: F = (K * (I1 + I2) ^ 1.5) / (r ^ 3)"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current1 + current2) ** (1.5)) / (distance ** 3)

# --- v1 laws ---
def _ground_truth_law_easy_v1(current1: float, current2: float, distance: float) -> float:
    """Easy law: F = (K * (I1 * I2) ^ 2) / r"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current1 * current2) ** (2)) / distance

def _ground_truth_law_medium_v1(current1: float, current2: float, distance: float) -> float:
    """Medium law: F = (K * (I1 * I2) ^ 2) * r"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current1 * current2) ** (2)) * distance

def _ground_truth_law_hard_v1(current1: float, current2: float, distance: float) -> float:
    """Hard law: F = (K * (I1 - I2) ^ 2) * r"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current1 - current2) ** (2)) * distance

# --- v2 laws ---
def _ground_truth_law_easy_v2(current1: float, current2: float, distance: float) -> float:
    """Easy law: F = (K * I2) / r"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * current2) / distance

def _ground_truth_law_medium_v2(current1: float, current2: float, distance: float) -> float:
    """Medium law: F = (K * I2) / r ^ 3.8"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * current2) / (distance ** 3.8)

def _ground_truth_law_hard_v2(current1: float, current2: float, distance: float) -> float:
    """Hard law: F = (K * (I2 ^ 0.9)) / r ^ 3.8"""
    if distance <= 0 or current1 == 0 or current2 == 0:
        return 0.0
    return (MAGNETIC_CONSTANT_K * (current2 ** 0.9)) / (distance ** 3.8)

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
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}. Choose from 'easy', 'medium', 'hard'.")
    available_versions = list(LAW_REGISTRY[difficulty].keys())
    if law_version is None:
        selected_version = random.choice(available_versions)
    else:
        if law_version not in available_versions:
            raise ValueError(f"Invalid law version '{law_version}' for difficulty '{difficulty}'. Available: {available_versions}")
        selected_version = law_version
    return LAW_REGISTRY[difficulty][selected_version], selected_version

def get_available_law_versions(difficulty: str) -> List[str]:
    """Get list of available law versions for a difficulty level."""
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    return list(LAW_REGISTRY[difficulty].keys())