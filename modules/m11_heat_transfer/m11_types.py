from typing import Dict, Any
from modules.common.types import ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

# Only essential constants for heat transfer distribution
HEAT_TRANSFER_CONSTANTS = {
    'MIN_DISTRIBUTION': 0.1,                # Minimum fraction for any heat transfer mechanism
    'MAX_DISTRIBUTION': 0.8,                # Maximum fraction for any heat transfer mechanism
    'DEFAULT_NOISE_LEVEL': 0.01,            # Default noise level for measurements
    'TIME_SCALING_FACTOR': 100.0,           # Scaling factor for time calculation: t = (m * c) / 100
    'MIN_ENERGY_LOSS': 0.18,                # Minimum energy loss before distribution (18%)
    'MAX_ENERGY_LOSS': 0.22,                # Maximum energy loss before distribution (22%)
    'LIGHT_BULB_POWER': 1.0,                # Power required per light bulb in Watts
    'MIN_SPECIFIC_HEAT_FACTOR': 0.7,        # Minimum factor to multiply c by for alternative calculation
    'MAX_SPECIFIC_HEAT_FACTOR': 1.2,        # Maximum factor to multiply c by for alternative calculation
}

# Precision constants for evaluation
ABSOLUTE_HEAT_TRANSFER_PRECISION = 1e-9
ABSOLUTE_POWER_PRECISION = 1e-9
