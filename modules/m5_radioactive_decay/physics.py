"""
Physics module for m5_radioactive_decay.
This module contains physics-related functions and constants for radioactive decay experiments.
"""

import numpy as np
from typing import Tuple, List

def calculate_decay_constant(half_life: float) -> float:
    """
    Calculate decay constant from half-life.
    
    Args:
        half_life: Half-life of the radioactive isotope
        
    Returns:
        Decay constant (λ = ln(2) / half_life)
    """
    return np.log(2) / half_life

def calculate_activity(
    initial_activity: float,
    decay_constant: float,
    time: float
) -> float:
    """
    Calculate current activity using exponential decay law.
    
    Args:
        initial_activity: Initial activity (N₀)
        decay_constant: Decay constant (λ)
        time: Time elapsed since initial measurement
        
    Returns:
        Current activity (N = N₀ * e^(-λt))
    """
    return initial_activity * np.exp(-decay_constant * time)

def calculate_remaining_atoms(
    initial_atoms: float,
    decay_constant: float,
    time: float
) -> float:
    """
    Calculate remaining number of atoms using exponential decay law.
    
    Args:
        initial_atoms: Initial number of atoms (N₀)
        decay_constant: Decay constant (λ)
        time: Time elapsed since initial measurement
        
    Returns:
        Remaining number of atoms (N = N₀ * e^(-λt))
    """
    return initial_atoms * np.exp(-decay_constant * time)

def calculate_half_life_from_decay_constant(decay_constant: float) -> float:
    """
    Calculate half-life from decay constant.
    
    Args:
        decay_constant: Decay constant (λ)
        
    Returns:
        Half-life (t₁/₂ = ln(2) / λ)
    """
    return np.log(2) / decay_constant

def calculate_mean_lifetime(decay_constant: float) -> float:
    """
    Calculate mean lifetime from decay constant.
    
    Args:
        decay_constant: Decay constant (λ)
        
    Returns:
        Mean lifetime (τ = 1 / λ)
    """
    return 1.0 / decay_constant

def calculate_decay_rate(
    current_atoms: float,
    decay_constant: float
) -> float:
    """
    Calculate instantaneous decay rate.
    
    Args:
        current_atoms: Current number of atoms
        decay_constant: Decay constant (λ)
        
    Returns:
        Decay rate (dN/dt = -λN)
    """
    return -decay_constant * current_atoms

def calculate_time_for_activity_reduction(
    initial_activity: float,
    final_activity: float,
    decay_constant: float
) -> float:
    """
    Calculate time required for activity to reduce from initial to final value.
    
    Args:
        initial_activity: Initial activity
        final_activity: Final activity
        decay_constant: Decay constant (λ)
        
    Returns:
        Time required (t = ln(N₀/N) / λ)
    """
    if final_activity <= 0 or initial_activity <= 0:
        return float('inf')
    return np.log(initial_activity / final_activity) / decay_constant

