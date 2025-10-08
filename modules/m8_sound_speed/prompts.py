from .m8_types import ExperimentSystem
from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- gamma: adiabatic index of the gas. It should be a positive real number.
- T: temperature of the medium in Kelvin. It should be a positive real number.
- M: molar mass of the medium. It should be a positive real number."""

# --- Core Law Discovery ---
FUNCTION_SIGNATURE = "def discovered_law(gamma, T, M):"
RETURN_DESCRIPTION = "the speed of sound (v)"
EXAMPLE = """**Example:**
<final_law>
def discovered_law(gamma, T, M):
    import math
    R = 1.143
    return math.sqrt(gamma * R * T / M)
</final_law>

**Note**:
- `gamma` is the adiabatic index of the gas.
- `R` is the ideal gas constant (value may be different in the simulated universe).
- `T` is the temperature in Kelvin.
- `M` is the molar mass of the gas in kg/mol.
- If possible, group and wrap all variables into a single expression when appropriate (e.g., instead of T * sqrt(M), prefer sqrt(T² * M) to express the same relationship in a more unified form)."""

VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a device that can directly measure the speed of sound in various ideal gases under controlled conditions.
**Control Parameters:**
- `gamma`: The adiabatic index of the gas.
- `T`: The temperature of the gas in Kelvin.
- `M`: The molar mass of the gas in kg/mol.

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"gamma": ..., "T": ..., "M": ...}},
  {{"gamma": ..., "T": ..., "M": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measured sound speeds (v).
<experiment_output>
[..., ...]
</experiment_output>"""

# --- Simple System: Echo Method ---
SIMPLE_SYSTEM_PROMPT = """**Experimental Apparatus:**
You are using an echo-based setup to determine the speed of sound. A sound pulse is emitted towards a wall at a known distance, and the time it takes for the echo to return is measured.

**Experimental Setup:**
1.  **Gas Chamber**: A chamber filled with a specific gas. You can control the gas properties.
2.  **Sound Emitter/Detector**: Emits a sound pulse and records the time until the echo is detected.
3.  **Movable Wall**: A wall placed at a known distance from the emitter.

**Control Parameters:**
- `adiabatic_index`: The adiabatic index (`gamma`) of the gas in the chamber.
- `molar_mass`: The molar mass (`M`) of the gas in kg/mol.
- `temperature`: The temperature (`T`) of the gas in Kelvin.
- `distance`: The distance (`d`) to the wall in meters.

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"adiabatic_index": ..., "molar_mass": ..., "temperature": ..., "distance": ...}},
  {{"adiabatic_index": ..., "molar_mass": ..., "temperature": ..., "distance": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the total time taken for the echo to return.
<experiment_output>
[
  {{"time": "..."}},
  {{"time": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Wave Speed Equation:** `v = f * λ`
    - The speed of any wave (`v`) is the product of its frequency (`f`) and its wavelength (`λ`)
    
2.  **Sound Intensity:** `I = P / (4πr²)`
    - The intensity (`I`) of a sound wave, which relates to its loudness, decreases with the square of the distance (`r`) from the source. `P` is the power of the source.

3.  **Doppler Effect:** `f' = f * (v ± v_o) / (v ∓ v_s)`
    - This law describes the change in the observed frequency of a wave (`f'`) when there is relative motion between the source (speed `v_s`) and the observer (speed `v_o`). The original frequency is `f` and the wave speed is `v`."""

# --- Complex System: Resonance Tube ---
COMPLEX_SYSTEM_PROMPT = """**Experimental Apparatus:**
You are using a resonance tube to find the speed of sound. A speaker generating a sound of a known frequency is placed at the open end of a tube filled with a specific gas. The length of the air column inside the tube can be changed. You record the lengths at which resonance (maximum loudness) occurs.

**Experimental Setup:**
1.  **Resonance Tube**: A tube with a specific internal `tube_diameter`, open at the speaker end and closed by a movable piston. The gas in the tube can be controlled.
2.  **Speaker**: Generates a sound wave of a fixed, known frequency.
3.  **Measurement**: You record the first two lengths of the air column (`L₁` and `L₂`) in meters where resonance occurs.

**Control Parameters:**
- `adiabatic_index`: The adiabatic index (`gamma`) of the gas in the tube.
- `molar_mass`: The molar mass (`M`) of the gas in kg/mol.
- `temperature`: The temperature (`T`) of the gas in Kelvin.
- `driving_frequency`: The frequency (`f`) of the sound wave from the speaker in Hertz.
- `tube_diameter`: The internal diameter of the resonance tube in meters.

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"adiabatic_index": ..., "molar_mass": ..., "temperature": ..., "driving_frequency": ..., "tube_diameter": ...}},
  {{"adiabatic_index": ..., "molar_mass": ..., "temperature": ..., "driving_frequency": ..., "tube_diameter": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the first two resonance lengths (in m).
<experiment_output>
[
  {{"first_resonance_length": "...", "second_resonance_length": "..."}},
  {{"first_resonance_length": "...", "second_resonance_length": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Wave Speed Equation:** `v = f * λ`
    - The speed of any wave (`v`) is the product of its frequency (`f`) and its wavelength (`λ`).

2.  **Resonance Condition**: 
    - In a tube open at one end, resonance occurs when the *effective length* of the air column is an odd multiple of a quarter-wavelength (e.g., `λ/4`, `3λ/4`, `5λ/4`, etc.).

3.  **End Correction**:
    - The antinode of the standing wave forms slightly outside the open end of the tube. This means the effective length is the measured length (`L`) plus an 'end correction' factor (`e`).

4.  **End Correction and Diameter**:
    - The end correction factor (`e`) is proportional to the `tube_diameter` (`d`). The exact relationship is `e = 0.3 * d`."""

# Code assisted specific instructions for interactive Python code execution
CODE_ASSISTED_PROMPT_INSTRUCTION = """**IMPORTANT: You have access to interactive Python code execution through <python> tags.**

**How to use <python> tags:**
1. **You can write ANY Python code** - functions, calculations, print statements, etc.
2. **Format:** <python>your_python_code_here</python>
3. **You can submit multiple <python> tags** to test different ideas
4. **Each <python> tag will be executed** and you'll get immediate feedback in <python_output> tags
5. **Use the feedback** to refine your understanding and calculations

**CRITICAL: Use EXACTLY these tags:**
- Start with: `<python>`
- End with: `</python>`
- NOT `<|python_tag|>` or any other variation

**Examples:**

**Complex calculation:**
```
<python>
import math
from scipy.integrate import quad

# Example involving logarithms and exponentials
x = 2.5
log_val = math.log(x)
exp_val = math.exp(x)
result = log_val * exp_val
print(f"ln({x}) * e^{x} = {result}")

# Example involving definite integrals
def f(x):
    return math.exp(-x**2)

# Integrate from 0 to 1
result, error = quad(f, 0, 1)

print("Integral result:", result)
print("Estimated error:", error)
</python>
```

**Function definition and testing:**
```
<python>
def test_hypothesis(gamma, T, M):
    import math
    R = 1.1413
    try:
        return math.sqrt(gamma * R * T / M)
    except (ValueError, ZeroDivisionError):
        return float('nan')

# Test with parameters for air at 300K
# gamma = 1.4, T = 300 K, M = 0.02897 kg/mol
print(test_hypothesis(1.4, 300, 0.02897))
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
128.632

**Your Code:**
```python
def test_hypothesis(gamma, T, M):
    import math
    R = 1.1413
    try:
        return math.sqrt(gamma * R * T / M)
    except (ValueError, ZeroDivisionError):
        return float('nan')

# Test with parameters for air at 300K
# gamma = 1.4, T = 300 K, M = 0.02897 kg/mol
print(test_hypothesis(1.4, 300, 0.02897))
```
</python_output>
```

**Data-driven discovery of exponents:**
```
<python>
import pandas as pd
import numpy as np

# Sample experimental data collected from the simulation
data = {
    'v': [347.2, 1020.5, 344.4, 377.8, 1105.8],
    'gamma': [1.4, 1.67, 1.3, 1.4, 1.67],
    'T': [300, 300, 350, 350, 350],
    'M': [0.02897, 0.004, 0.032, 0.02897, 0.004]
}
df = pd.DataFrame(data)

# We suspect a power-law relationship: v = C * gamma^a * T^b * M^c
# To find the exponents a, b, c, we can use linear regression on the log-transformed equation:
# log(v) = log(C) + a*log(gamma) + b*log(T) + c*log(M)

df_log = np.log(df)

# Prepare the matrix for linear regression
# The columns of A correspond to log(gamma), log(T), and log(M)
A = df_log[['gamma', 'T', 'M']].values
y = df_log['v'].values

# Solve for the coefficients [a, b, c]
# We add a column of ones to A to solve for the intercept log(C) as well
A_with_intercept = np.hstack([A, np.ones((A.shape[0], 1))])
coeffs, _, _, _ = np.linalg.lstsq(A_with_intercept, y, rcond=None)

a, b, c = coeffs[0], coeffs[1], coeffs[2]

print(f"Found exponents (a, b, c): {a:.2f}, {b:.2f}, {c:.2f}")
print("This suggests the law is v ∝ gamma^0.50 * T^0.50 * M^-0.50")
</python>
```

**Workflow:**
1. **Analyze the problem** and form initial hypotheses
2. **Use <python> tags** to test your ideas, perform calculations, or explore data
3. **Analyze the results** from <python_output> and refine your understanding
4. **Repeat** with more <python> tags until you're confident in your solution
5. **Submit final law** using <final_law> tags with proper Python function format"""

def get_task_prompt(system: str, is_code_assisted: bool = False, noise_level: float = 0.0) -> str:
    """Return the appropriate task prompt based on system."""
    prompts = [OBJECTIVE_PROMPT]

    if noise_level > 0.0:
        run_experiment_instruction = RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
    else:
        run_experiment_instruction = RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE
    
    if system == ExperimentSystem.VANILLA_EQUATION:
        prompts.append(VANILLA_EQUATION_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        prompts.append(SIMPLE_SYSTEM_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction, ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER))
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        prompts.append(COMPLEX_SYSTEM_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction, ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER))
    
    if is_code_assisted:
        prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
    
    prompts.append(SUBMISSION_REQUIREMENTS.format(
        function_signature=FUNCTION_SIGNATURE,
        return_description=RETURN_DESCRIPTION,
        example=EXAMPLE
    ))
    
    return "\n\n".join(prompts)
