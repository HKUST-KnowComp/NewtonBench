from typing import Callable, Tuple, List, Optional
import math
import random
import numpy as np

# --- Environment Constants ---
HIDDEN_CONSTANT_C = 1.0513e-14

# --- Ground Truth Laws ---

# --- v0 laws ---
def _ground_truth_law_easy_v0(omega: float, T: float) -> float:
    """Easy law: n = 1 / (exp(C * ω / T) + 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * omega / T) + 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_medium_v0(omega: float, T: float) -> float:
    """Medium law: n = 1 / (exp(C * ω ^ 1.5 / T) + 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**1.5) / T) + 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_hard_v0(omega: float, T: float) -> float:
    """Hard law: n = 1 / (exp(C * ω ^ 1.5 / T ^ 2) + 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**1.5) / (T**2)) + 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')
    
# --- v1 laws ---
def _ground_truth_law_easy_v1(omega: float, T: float) -> float:
    """Easy law: n = 1 / (exp(C * ω ^ 0.5 / T) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**0.5) / T) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_medium_v1(omega: float, T: float) -> float:
    """Medium law: n = 1 / (exp(C * ω ^ 0.5 * T) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**0.5) * T) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_hard_v1(omega: float, T: float) -> float:
    """Hard law: n = 1 / (exp(C * (ω ^ 0.5 * T ^ 2.3)) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**0.5 * T**2.3)) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')
    
# --- v2 laws ---
def _ground_truth_law_easy_v2(omega: float, T: float) -> float:
    """Easy law: n = 1 / (exp(C * ω / T ^ 3) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * omega / (T ** 3)) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_medium_v2(omega: float, T: float) -> float:
    """Medium law: n = 1 / (exp(C * ω ^ 1.5 / T ^ 3) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (np.exp(HIDDEN_CONSTANT_C * (omega**1.5) / (T ** 3)) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return float('nan')

def _ground_truth_law_hard_v2(omega: float, T: float) -> float:
    """Hard law: n = 1 / (-ln(C * ω ^ 1.5 / T ^ 3) - 1)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = 1 / (-np.log(HIDDEN_CONSTANT_C * (omega**1.5) / (T ** 3)) - 1)
        if value > 0 and np.isfinite(value):
            return value
        else:
            return float('nan')
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
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
