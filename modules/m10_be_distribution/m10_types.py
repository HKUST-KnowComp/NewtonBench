from modules.common.types import ExperimentSystem

# Precision constants
ABSOLUTE_SPECTRAL_RADIANCE_PRECISION = 1e-12
ABSOLUTE_POWER_PRECISION = 1e-12
ABSOLUTE_OCCUPATION_NUMBER_PRECISION = 1e-18

# Default parameters for the Black-Body Spectrometer experiment
BLACK_BODY_DEFAULTS = {
    'temperature': 1e3, 
    'probe_frequency': 1e8 
}

# Default parameters for the complex system experiment
DIFFICULT_MODEL_DEFAULTS = {
    'temperature': 1e3, 
    'center_frequency': 1e9,
    'bandwidth': 1e1
}
