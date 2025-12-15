from modules.common.types import ExperimentSystem, ABSOLUTE_POSITION_PRECISION, ABSOLUTE_VELOCITY_PRECISION, ABSOLUTE_FORCE_PRECISION

ABSOLUTE_MAGNETIC_FIELD_PRECISION = 1e-10

LINEAR_DEFAULTS = {
    'time_step': 0.05,
    'duration': 1.0,
    'initial_velocity': 0.0,
    'mass_wire': 1.0, 
}

# Default parameters for fixed wire experiments (complex system)
FIXED_WIRE_DEFAULTS = {
    'current1': 1.0,  # AC amplitude for wire 1
    'current2': 1.0,  # DC current for wire 2
    'mass_wire': 0.01,  # Mass of the moving wire
    'distance': 0.1,  # Initial distance between wires
    'initial_velocity': 0.0,  # Initial velocity of the moving wire
}