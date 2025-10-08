from typing import Dict, Any
from modules.common.types import ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

# Default parameters for different experiment modes
HOOKE_DEFAULTS = {
    'default_mass': 1.0,                    # Default mass in kg
    'default_displacement_scale': 0.02,     # Characteristic displacement scale for energy decay
    'num_points': 20,                       # Default number of time points
}

ENERGY_DEFAULTS = {
    'num_points': 20,
}

# Precision constants for evaluation
ABSOLUTE_ENERGY_PRECISION = 1e-9
