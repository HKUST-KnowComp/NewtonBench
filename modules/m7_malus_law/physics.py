import numpy as np


def calculate_transmitted_intensity(I_0: float, theta: float, malus_law: callable) -> float:
    if malus_law is None:
        raise ValueError("malus_law must be provided")
    return malus_law(I_0, theta)


def calculate_intensity_at_angle(I_0: float, theta: float, malus_law: callable) -> float:
    return calculate_transmitted_intensity(I_0, theta, malus_law)


def calculate_angle_for_intensity(I_0: float, target_intensity: float, malus_law: callable) -> float:
    if target_intensity <= 0 or I_0 <= 0:
        return float('inf')
    ratio = target_intensity / I_0
    if ratio > 1:
        return 1e-6
    elif ratio < 0:
        return np.pi / 2
    angle = np.arccos(np.sqrt(ratio))
    return max(1e-6, angle)


def calculate_intensity_ratio(I_0: float, theta1: float, theta2: float, malus_law: callable) -> float:
    I1 = calculate_transmitted_intensity(I_0, theta1, malus_law)
    I2 = calculate_transmitted_intensity(I_0, theta2, malus_law)
    if I2 == 0:
        return float('inf')
    return I1 / I2


def calculate_polarization_efficiency(I_0: float, theta: float, malus_law: callable) -> float:
    transmitted = calculate_transmitted_intensity(I_0, theta, malus_law)
    return transmitted / I_0 if I_0 > 0 else 0.0


