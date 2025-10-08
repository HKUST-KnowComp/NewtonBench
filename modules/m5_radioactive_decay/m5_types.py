from typing import Dict, Any
from modules.common.types import ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

# Default parameters for different experiment modes
LINEAR_DEFAULTS = {
    'num_points': 20,
}

TWO_DIM_DEFAULTS = {
    'num_points': 20,
}

# Precision constants for evaluation
ABSOLUTE_ACTIVITY_PRECISION = 1e-9
ABSOLUTE_RATIO_PRECISION = 1e-12