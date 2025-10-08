from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """- k: thermal conductivity coefficient of the material. It should be a positive real number.
- A: cross-sectional area. It should be a positive real number.
- delta_T: temperature difference. It can be assumed to be a positive real number.
- d: thickness of the material. It should be a positive real number."""

# Fourier law-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(k, A, delta_T, d):"
RETURN_DESCRIPTION = "the power transfer rate (P) between two regions with temperature difference"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(k, A, delta_T, d):
   import math
   return (k * A * delta_T) / (d ** 3)
</final_law>

**Example 2:**
<final_law>
def discovered_law(k, A, delta_T, d):
   return (k * A * (delta_T ** 2.5)) / (d ** 3)
</final_law>

**Note:** 
- k is the thermal conductivity coefficient
- A is the cross-sectional area
- delta_T is the temperature difference (can be assumed to be > 0)
- d is the distance/thickness (always positive)"""

# Vanilla equation discovery prompt for Fourier law
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a thermal conductivity measurement device that can measure power transfer between two regions with different temperatures. You have precise control over the following properties for each experiment you run:
- Thermal conductivity coefficient (`k`) - always positive
- Cross-sectional area (`A`) - always positive
- Temperature difference (`delta_T`) - can be assumed to be > 0
- Distance/thickness (`d`) - always positive

**Important Notes:**
- Cross-sectional area `A` is always positive and represents the area through which heat flows
- Temperature difference `delta_T` can be assumed to be > 0, representing the magnitude of temperature difference
- Distance/thickness `d` is always positive and represents the separation between the two regions
- The power transfer rate P represents the rate of heat energy transfer per unit time

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat flow in the same way (e.g., both material properties or both driving forces)
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls driving force, another controls resistance)

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

*Your Request:*
<run_experiment>
[
  {{"k": ..., "A": ..., "delta_T": ..., "d": ...}},
  {{"k": ..., "A": ..., "delta_T": ..., "d": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured power transfer rate
<experiment_output>
[1.234e-05, 2.345e-04]
</experiment_output>"""

# Simple system discovery prompt (skeleton for future implementation)
SIMPLE_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a 1D thermal conduction system that can:
1. Control thermal conductivity, area, temperature difference, and distance
2. Track temperature profiles across spatial positions (0 to d)
3. Measure temperature decay patterns along the conduction path

**Control Parameters:**
- `k`: Thermal conductivity coefficient (always positive)
- `A`: Cross-sectional area (always positive)
- `delta_T`: Temperature difference (can be assumed to be > 0)
- `d`: Distance/thickness (always positive)
- `num_points`: Number of spatial points to sample

**Important Notes:**
- This is a simplified 1D thermal conduction model
- The system provides spatial temperature profile data across positions
- Heat flows from higher temperature to lower temperature regions
- The power transfer rate P is related to the heat flux through the material

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"k": ..., "A": ..., "delta_T": ..., "d": ..., "num_points": ...}},
   {{"k": ..., "A": ..., "delta_T": ..., "d": ..., "num_points": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of spatial temperature profile data objects (at most 20 data points per experiment):
<experiment_output>
[
   {{"x": [...], "T": [...]}} ,
   {{"x": [...], "T": [...]}}
]
</experiment_output>

**Physics Background:**
- Heat conduction follows Fourier's law of heat conduction
- Temperature gradients drive heat flow
- The system tracks temperature profiles across spatial positions

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat flow in the same way (e.g., both material properties or both driving forces)
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls driving force, another controls resistance)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Heat energy is conserved in the system
   - Heat cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Thermal Diffusion: Temperature changes propagate through the material
   - Heat transfer occurs through molecular interactions
   - Temperature profiles evolve according to the heat equation

3. Temperature Profile Physics:
   - Temperatures decay linearly, not exponentially: T(x) = T0 - (P/(k*A))*x"""

# Complex system discovery prompt (1D heat flux measurement)
COMPLEX_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a 1D thermal conduction system that can:
1. Control thermal conductivity, area, temperature difference, and distance
2. Track temperature profiles across spatial positions (0 to d)
3. Measure heat flux at different positions using Fourier's law
4. Calculate heat flux from temperature gradients

**Control Parameters:**
- `k`: Thermal conductivity coefficient (always positive)
- `A`: Cross-sectional area (always positive)
- `delta_T`: Temperature difference (can be assumed to be > 0)
- `d`: Distance/thickness (always positive)
- `num_points`: Number of spatial points to sample

**Important Notes:**
- This is a 1D thermal conduction model that measures heat flux
- The system provides spatial heat flux data across positions
- Heat flux is calculated using the temperature gradient at each point
- The system uses finite difference approximation for gradient calculation

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"k": ..., "A": ..., "delta_T": ..., "d": ..., "num_points": ...}},
   {{"k": ..., "A": ..., "delta_T": ..., "d": ..., "num_points": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of spatial data objects:
<experiment_output>
[
   {{"x": [...], "heat_flux": [...]}} ,
   {{"x": [...], "heat_flux": [...]}}
]
</experiment_output>

**Physics Background:**
- 1D heat conduction with heat flux measurement
- Heat flux is calculated at each spatial position
- The system uses finite difference methods for gradient calculation
- Heat flux patterns reveal the underlying power law governing heat transfer

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat flow in the same way (e.g., both material properties or both driving forces)
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls driving force, another controls resistance)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Heat energy is conserved in the system
   - Heat cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Thermal Diffusion: Temperature changes propagate through the material
   - Heat transfer occurs through molecular interactions
   - Temperature profiles evolve according to the heat equation

3. Heat Flux Calculation: Finite difference approximation
   - Heat flux is derived from spatial temperature variations
   - The system measures how temperature changes across positions
   - Heat flux patterns reveal the underlying heat transfer law
   
4. Temperature Profile Physics:
   - Temperatures decay exponentially"""

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
def test_hypothesis(k, A, delta_T, d):
    # Test your hypothesis: P = (k * A * delta_T) / d^2
    return (k * A * delta_T) / (d ** 2)

# Test with different parameters
test_k = [1.0, 2.0, 3.0]
test_A = [1.0, 1.0, 1.0]
test_delta_T = [1.0, 2.0, 3.0]
test_d = [1.0, 2.0, 3.0]

for k_val, A_val, delta_T_val, d_val in zip(test_k, test_A, test_delta_T, test_d):
    power = test_hypothesis(k_val, A_val, delta_T_val, d_val)
    print(f"k={k_val}, A={A_val}, ΔT={delta_T_val}, d={d_val} → P={power}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
k=1.0, A=1.0, ΔT=1.0, d=1.0 → P=1.0
k=2.0, A=1.0, ΔT=2.0, d=2.0 → P=1.0
k=3.0, A=1.0, ΔT=3.0, d=3.0 → P=1.0

**Your Code:**
```python
def test_hypothesis(k, A, delta_T, d):
    # Test your hypothesis: P = (k * A * delta_T) / d^2
    return (k * A * delta_T) / (d ** 2)

# Test with different parameters
test_k = [1.0, 2.0, 3.0]
test_A = [1.0, 1.0, 1.0]
test_delta_T = [1.0, 2.0, 3.0]
test_d = [1.0, 2.0, 3.0]

for k_val, A_val, delta_T_val, d_val in zip(test_k, test_A, test_delta_T, test_d):
    power = test_hypothesis(k_val, A_val, delta_T_val, d_val)
    print(f"k={k_val}, A={A_val}, ΔT={delta_T_val}, d={d_val} → P={power}")
```
</python_output>
```

**Workflow:**
1. **Analyze the problem** and form initial hypotheses
2. **Use <python> tags** to test your ideas, perform calculations, or explore data
3. **Analyze the results** from <python_output> and refine your understanding
4. **Repeat** with more <python> tags until you're confident in your solution
5. **Submit final law** using <final_law> tags with proper Python function format"""

def get_task_prompt(system: str, is_code_assisted: bool = False, noise_level: float = 0.0) -> str:
    """
    Return the appropriate task prompt based on system.
    Args:
       system: string, one of "vanilla_equation", "simple_system", "complex_system"
       is_code_assisted: boolean, if True includes code assisted instructions
    Returns:
       Complete task prompt string
    """
    prompts = [OBJECTIVE_PROMPT]

    if noise_level > 0.0:
        run_experiment_instruction = RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
    else:
        run_experiment_instruction = RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE

    if system == ExperimentSystem.VANILLA_EQUATION:
        prompts.append(VANILLA_EQUATION_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION = run_experiment_instruction))
    elif system == ExperimentSystem.SIMPLE_SYSTEM:
        prompts.append(SIMPLE_SYSTEM_DISCOVERY_PROMPT.format(ASSISTING_LAWS_DISCLAIMER = ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION = run_experiment_instruction))
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        prompts.append(COMPLEX_SYSTEM_DISCOVERY_PROMPT.format(ASSISTING_LAWS_DISCLAIMER = ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION = run_experiment_instruction))

    # Add code assisted instructions if requested
    if is_code_assisted:
        prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
    
    prompts.append(SUBMISSION_REQUIREMENTS.format(
        function_signature = FUNCTION_SIGNATURE,
        return_description = RETURN_DESCRIPTION,
        example = EXAMPLE
    ))
    return "\n\n".join(prompts)
