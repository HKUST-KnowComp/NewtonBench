from enum import Enum

class ExperimentSystem(str, Enum):
    """Experiment system enumeration (shared by all modules)."""
    VANILLA_EQUATION = "vanilla_equation" 
    SIMPLE_SYSTEM = "simple_system"
    COMPLEX_SYSTEM = "complex_system"

# Measurement precision constants (shared)
ABSOLUTE_POSITION_PRECISION = 1e-9  # meters
ABSOLUTE_VELOCITY_PRECISION = 1e-9  # m/s
ABSOLUTE_FORCE_PRECISION = 1e-12     # N 