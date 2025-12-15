from modules.common.types import ExperimentSystem, ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

# Precision constants
ABSOLUTE_ANGULAR_VELOCITY_PRECISION = 1e-9
ABSOLUTE_PERIOD_PRECISION = 1e-9
ABSOLUTE_AMPLITUDE_PRECISION = 1e-9

# Default parameters for the damped oscillator experiment
DAMPED_OSCILLATOR_DEFAULTS = {
    'k_constant': 10.0,
    'mass': 1.0,
    'b_constant': 0.5,
    'initial_amplitude': 1.0,
}
