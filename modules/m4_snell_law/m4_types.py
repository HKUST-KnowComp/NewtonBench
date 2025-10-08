from modules.common.types import ExperimentSystem, ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

# Constants for Snell's law precision
ABSOLUTE_ANGLE_PRECISION = 1e-6

# Speed of light constant (m/s)
SPEED_OF_LIGHT = 3.0e8

# Default parameters for light propagation experiments (simple system)
LIGHT_PROPAGATION_DEFAULTS = {
    'speed_medium1': 3.0e8,  # Speed of light in vacuum
    'speed_medium2': 2.0e8,  # Example speed in a medium
    'incidence_angle': 30.0  # degrees
}

# Default parameters for triple-layer experiments (complex system)
TRIPLE_LAYER_DEFAULTS = {
    'refractive_index_1': 1.0,  # vacuum
    'refractive_index_2': 1.5,  # glass
    'refractive_index_3': 1.33,  # water
    'incidence_angle': 30.0,  # degrees
}