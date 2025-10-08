import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_heat_transfer(m: float, c: float, delta_T: float) -> float:
    """
    Calculate heat transfer using the basic formula Q = m * c * delta_T.
    
    Args:
        m: Mass in kg
        c: Specific heat capacity in J/(kg·K)
        delta_T: Temperature change in K
        
    Returns:
        Heat transfer in Joules
    """
    return m * c * delta_T

def calculate_specific_heat_capacity(material: str) -> float:
    """
    Get specific heat capacity for common materials.
    
    Args:
        material: Material name
        
    Returns:
        Specific heat capacity in J/(kg·K)
    """
    specific_heats = {
        'water': 4186.0,      # J/(kg·K)
        'aluminum': 900.0,    # J/(kg·K)
        'copper': 385.0,      # J/(kg·K)
        'iron': 450.0,        # J/(kg·K)
        'steel': 500.0,       # J/(kg·K)
        'glass': 840.0,       # J/(kg·K)
        'wood': 1700.0,       # J/(kg·K)
        'air': 1005.0,        # J/(kg·K)
    }
    
    return specific_heats.get(material.lower(), 1000.0)  # Default value

def calculate_temperature_change(Q: float, m: float, c: float) -> float:
    """
    Calculate temperature change from heat transfer.
    
    Args:
        Q: Heat transfer in Joules
        m: Mass in kg
        c: Specific heat capacity in J/(kg·K)
        
    Returns:
        Temperature change in K
    """
    if m <= 0 or c <= 0:
        return 0.0
    return Q / (m * c)

def calculate_mass_from_heat_transfer(Q: float, c: float, delta_T: float) -> float:
    """
    Calculate mass from heat transfer and temperature change.
    
    Args:
        Q: Heat transfer in Joules
        c: Specific heat capacity in J/(kg·K)
        delta_T: Temperature change in K
        
    Returns:
        Mass in kg
    """
    if c <= 0 or delta_T == 0:
        return 0.0
    return Q / (c * delta_T)

def generate_heat_transfer_profile(time_points: List[float], m: float, c: float, delta_T: float, heat_rate: float = 1.0) -> List[float]:
    """
    Generate heat transfer profile over time.
    
    Args:
        time_points: List of time points in seconds
        m: Mass in kg
        c: Specific heat capacity in J/(kg·K)
        delta_T: Temperature change in K
        heat_rate: Rate of heat transfer (default 1.0)
        
    Returns:
        List of heat transfer values at each time point
    """
    Q_total = m * c * delta_T
    heat_profile = []
    
    for t in time_points:
        # Heat transfer increases over time until reaching the total
        Q_t = Q_total * (1 - np.exp(-heat_rate * t))
        heat_profile.append(Q_t)
    
    return heat_profile
