from typing import Tuple

def calculate_echo_time(distance: float, speed: float) -> float:
    """Calculates the time for a sound to travel to a wall and back."""
    if speed <= 0:
        return float('inf')
    return 2 * distance / speed

def calculate_resonance_lengths(speed: float, frequency: float, tube_diameter: float) -> Tuple[float, float]:
    """Calculates the first two resonance lengths in a tube open at one end."""
    if frequency <= 0 or speed <= 0:
        return (float('nan'), float('nan'))
    
    wavelength = speed / frequency
    
    # End correction factor
    end_correction = 0.3 * tube_diameter
    
    # Effective lengths for resonance are L + e
    # L1 + e = wavelength / 4  => L1 = wavelength / 4 - e
    # L2 + e = 3 * wavelength / 4 => L2 = 3 * wavelength / 4 - e
    
    L1 = wavelength / 4 - end_correction
    L2 = 3 * wavelength / 4 - end_correction
    
    # Physical lengths cannot be negative
    if L1 < 0:
        L1 = 0
    if L2 < 0:
        L2 = 0

    return L1, L2