import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_hooke_energy(k: float, x: float) -> float:
    """
    Calculate the elastic potential energy stored in a spring.
    
    Args:
        k: Spring constant in N/m
        x: Displacement from equilibrium in meters
        
    Returns:
        Elastic potential energy stored in Joules
    """
    return 0.5 * k * (x ** 2)

def calculate_energy_density(k: float, x: float, volume: float) -> float:
    """
    Calculate the energy density of a spring system.
    
    Args:
        k: Spring constant in N/m
        x: Displacement from equilibrium in meters
        volume: Volume of the spring system in cubic meters
        
    Returns:
        Energy density in Joules per cubic meter
    """
    energy = calculate_hooke_energy(k, x)
    return energy / volume if volume > 0 else 0.0

def calculate_equivalent_spring_constant(springs: List[float], connection: str = 'series') -> float:
    """
    Calculate equivalent spring constant for multiple springs.
    
    Args:
        springs: List of spring constant values in N/m
        connection: 'series' or 'parallel'
        
    Returns:
        Equivalent spring constant in N/m
    """
    if not springs:
        return 0.0
    
    if connection == 'series':
        # 1/keq = 1/k1 + 1/k2 + ... + 1/kn
        return 1.0 / sum(1.0 / k for k in springs if k > 0)
    elif connection == 'parallel':
        # keq = k1 + k2 + ... + kn
        return sum(k for k in springs if k > 0)
    else:
        raise ValueError("Connection must be 'series' or 'parallel'")

def calculate_displacement_distribution(springs: List[float], total_displacement: float, connection: str = 'series') -> List[float]:
    """
    Calculate displacement distribution across springs in series.
    
    Args:
        springs: List of spring constant values in N/m
        total_displacement: Total displacement across the system in meters
        connection: 'series' or 'parallel'
        
    Returns:
        List of displacement values across each spring
    """
    if connection == 'series':
        # In series, displacement divides inversely with spring constant
        total_spring_constant = calculate_equivalent_spring_constant(springs, 'series')
        return [total_displacement * (total_spring_constant / k) for k in springs if k > 0]
    elif connection == 'parallel':
        # In parallel, all springs have the same displacement
        return [total_displacement] * len(springs)
    else:
        raise ValueError("Connection must be 'series' or 'parallel'")

def generate_energy_profile(time_points: List[float], k: float, x: float, damping: float = 1.0) -> List[float]:
    """
    Generate energy profile over time for a damped spring oscillator.
    
    Args:
        time_points: List of time points in seconds
        k: Spring constant in N/m
        x: Initial displacement in meters
        damping: Damping coefficient in Ns/m
        
    Returns:
        List of energy values at each time point
    """
    omega = np.sqrt(k / 1.0)  # Assuming mass = 1 kg for simplicity
    energy_profile = []
    
    for t in time_points:
        # Energy as a function of time: E(t) = 0.5 * k * x^2 * e^(-damping * t)
        energy = 0.5 * k * (x ** 2) * np.exp(-damping * t)
        energy_profile.append(energy)
    
    return energy_profile
