from typing import Callable, Tuple, List, Optional
import math
import random

# --- Environment Constants ---
HIDDEN_CONSTANT = 3.516e2

# --- Ground Truth Laws ---

# --- v0 laws ---
def _ground_truth_law_easy_v0(gamma: float, T: float, M: float) -> float:
    """Easy law: v = sqrt(gamma * R * T**2 / M)"""
    try:
        return math.sqrt(gamma * HIDDEN_CONSTANT * T**2 / M)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v0(gamma: float, T: float, M: float) -> float:
    """Medium law: v = sqrt(gamma * R * T**2 / M**1.5)"""
    try:
        return math.sqrt(gamma * HIDDEN_CONSTANT * T**2 / M**1.5)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v0(gamma: float, T: float, M: float) -> float:
    """Hard law: v = sqrt((e ^ gamma) * R * T**2 / M**1.5)"""
    try:
        return math.sqrt((math.exp(gamma)) * HIDDEN_CONSTANT * T**2 / M**1.5)
    except (ValueError, ZeroDivisionError):
        return float('nan')
    
# --- v1 laws ---
def _ground_truth_law_easy_v1(gamma: float, T: float, M: float) -> float:
    """Easy law: v = gamma * R * T / M"""
    try:
        return gamma * HIDDEN_CONSTANT * T / M
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v1(gamma: float, T: float, M: float) -> float:
    """Medium law: v = gamma * T * R / (M ** 1/3)"""
    try:
        return gamma * T * HIDDEN_CONSTANT / (M ** (1/3))
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v1(gamma: float, T: float, M: float) -> float:
    """Hard law: v = ln(gamma) * T * R / (M ** 1/3)"""
    try:
        return math.log(gamma) * T * HIDDEN_CONSTANT / (M ** (1/3))
    except (ValueError, ZeroDivisionError):
        return float('nan')
    
# --- v2 laws ---
def _ground_truth_law_easy_v2(gamma: float, T: float, M: float) -> float:
    """Easy law: v = sqrt(R * T / M)"""
    try:
        return math.sqrt(HIDDEN_CONSTANT * T / M)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_medium_v2(gamma: float, T: float, M: float) -> float:
    """Medium law: v = sqrt(R * T * M ** 1.5)"""
    try:
        return math.sqrt(HIDDEN_CONSTANT * T * M ** 1.5)
    except (ValueError, ZeroDivisionError):
        return float('nan')

def _ground_truth_law_hard_v2(gamma: float, T: float, M: float) -> float:
    """Hard law: v = (R * T * M ** 1.5) ^ -2.8"""
    try:
        return (HIDDEN_CONSTANT * T * (M ** 1.5)) ** -2.8
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
    """
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    return list(LAW_REGISTRY[difficulty].keys())
