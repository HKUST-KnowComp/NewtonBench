from modules.common.types import ExperimentSystem, ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

ABSOLUTE_ENERGY_PRECISION = 1e-9

# Default parameters for 2D field simulation
TWO_DIM_DEFAULTS = {
    'time_step': 0.1,      # seconds
    'duration': 10.0,      # seconds
}

# Default parameters for 1D Coulomb experiment (linear motion)
LINEAR_DEFAULTS = {
    'time_step': 0.01,     # seconds
    'duration': 5.0,      # seconds
}
