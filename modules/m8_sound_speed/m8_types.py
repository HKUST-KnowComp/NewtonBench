from modules.common.types import ExperimentSystem, ABSOLUTE_VELOCITY_PRECISION

# Precision constants
ABSOLUTE_TIME_PRECISION = 1e-9
ABSOLUTE_LENGTH_PRECISION = 1e-9

# Default parameters for the echo method experiment (simple system)
ECHO_METHOD_DEFAULTS = {
    'adiabatic_index': 1.4,  
    'molar_mass': 0.02897,
    'temperature': 293.15, 
    'distance': 100.0,
}

# Default parameters for the resonance tube experiment (complex system)
RESONANCE_TUBE_DEFAULTS = {
    'adiabatic_index': 1.4,
    'molar_mass': 0.02897,
    'temperature': 293.15,
    'driving_frequency': 440.0,
    'tube_diameter': 0.1
}
