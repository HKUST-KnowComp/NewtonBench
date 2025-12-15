import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional

# --- Easy Difficulty Laws (v0 only) ---
def _ground_truth_law_easy_v0(N0: float, lambda_constant: float, t: float) -> float:
    """Easy radioactive decay law: N(t) = N₀ * e^(-λ * t^1.5)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 * np.exp(-lambda_constant * (t ** 1.5))
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_easy_v1(N0: float, lambda_constant: float, t: float) -> float:
    """Easy radioactive decay law: N(t) = N₀ * e^(-λ^1.5 * t)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 * np.exp(-lambda_constant ** 1.5 * t)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_easy_v2(N0: float, lambda_constant: float, t: float) -> float:
    """Easy radioactive decay law: N(t) = N₀ * e^(-(λ * t)^0.5)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 * np.exp(-(lambda_constant * t) ** 1.5)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

# --- Medium Difficulty Laws (v0 only) ---
def _ground_truth_law_medium_v0(N0: float, lambda_constant: float, t: float) -> float:
    """Medium radioactive decay law: N(t) = N₀^1.2 * e^(-λ * t^1.5)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.2 * np.exp(-lambda_constant * (t ** 1.5))
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v1(N0: float, lambda_constant: float, t: float) -> float:
    """Medium radioactive decay law: N(t) = N₀^1.4 * e^(-λ ** 1.5 * t)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.4 * np.exp(-lambda_constant ** 1.5 * t)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_medium_v2(N0: float, lambda_constant: float, t: float) -> float:
    """Medium radioactive decay law: N(t) = N₀^1.8 * e^(-(λ * t) ** 1.5)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.8 * np.exp(-(lambda_constant * t) ** 1.5)
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

# --- Hard Difficulty Laws (v0 only) ---
def _ground_truth_law_hard_v0(N0: float, lambda_constant: float, t: float) -> float:
    """Hard radioactive decay law: N(t) = N₀^1.2 * e^(-λ ** (np.exp(1)) * t^1.5)"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.2 * np.exp(-lambda_constant ** (np.exp(1)) * (t ** 1.5))
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v1(N0: float, lambda_constant: float, t: float) -> float:
    """Hard radioactive decay law: N(t) = N₀^1.4 * e^(-λ ** 1.5 * t ** (np.exp(1)))"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.4 * np.exp(-lambda_constant ** 1.5 * (t ** (np.exp(1))))
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
        return float('nan')

def _ground_truth_law_hard_v2(N0: float, lambda_constant: float, t: float) -> float:
    """Hard radioactive decay law: N(t) = N₀^1.8 * e^(-(λ * t) ** (np.exp(1) + 1.5))"""
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            value = N0 ** 1.8 * np.exp(-(lambda_constant * t) ** (np.exp(1) + 1.5))
        if not np.isfinite(value):
            return float('nan')
        return float(value)
    except FloatingPointError:
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