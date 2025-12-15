from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """- N0: initial number of atoms. It should be a positive real number.
- lambda_constant: It should be a positive real number.
- t: time elapsed. It should be a positive real number."""

# Radioactive decay-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(N0, lambda_constant, t):"
RETURN_DESCRIPTION = "the mathematical formula for the ground truth decay law that governs isotope decay in this universe"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(N0, lambda_constant, t):
   import math
   return N0 * math.exp(-lambda_constant * t)
</final_law>

**Example 2:**
<final_law>
def discovered_law(N0, lambda_constant, t):
   return N0 * math.exp(-lambda_constant * t)
</final_law>

**Note:** 
- N0 is the initial number of atoms
- lambda_constant (λ)
- t is the time elapsed since the initial measurement"""

# Vanilla Equation Prompt for radioactive decay
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a radioactive decay measurement device that can measure the current activity of radioactive samples. You have precise control over the following properties for each experiment you run:
- Initial Activity (`N0`) - always positive
- Lambda Constant (`lambda_constant`) - always positive
- Time (`t`) - always positive

**Important Notes:**
- Initial Activity `N0` is always positive and represents the starting activity of the sample
- Lambda Constant `lambda_constant` is always positive
- Time `t` is always positive and represents the time elapsed since initial measurement
- The current activity represents the remaining activity of the radioactive sample
- If your input results in overflowing the system, the system will return `NaN` (Not a Number)

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to decay behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls initial conditions, another controls time evolution)

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
  {{"N0": ..., "lambda_constant": ..., "t": ...}},
  {{"N0": ..., "lambda_constant": ..., "t": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured current activity.
<experiment_output>
[1.234e-05, 2.345e-04]
</experiment_output>"""

# Simple system discovery prompt for radiation detector radioactive decay
SIMPLE_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a radioactive sample placed in front of a radiation detector. The system can:
1. Control initial radioactive atom population and lambda constant
2. Track measured activity over time using the detector
3. Output how much remaining atoms left; however, it can only output 70% of the remaining atoms (i.e. measured activity is 0.7 * N(t))

**Control Parameters:**
- `N0`: Initial number of radioactive atoms (always positive)
- `lambda_constant`: Lambda constant (always positive)
- `t`: Time elapsed (always positive)
- `num_points`: Number of time points to sample

**Important Notes:**
- This is a radiation detector measurement model
- The system provides temporal measured activity data
- The observable is the "Measured Activity" over time, which is 70% of the remaining atoms
- If your input results in overflowing the system, the system will return `NaN` (Not a Number)

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"N0": ..., "lambda_constant": ..., "t": ..., "num_points": ...}},
   {{"N0": ..., "lambda_constant": ..., "t": ..., "num_points": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of temporal measured activity data objects (at most 20 data points per experiment):
<experiment_output>
[
   {{"time": [...], "measured_activity": [...]}} ,
   {{"time": [...], "measured_activity": [...]}}
]
</experiment_output>

**Important Data Handling Note:**
- The measured activity represents the detector's output over time
- All values are positive and represent the remaining atoms with 70% efficient (i.e. if N(t) is the remaining atom, then measured activity is 0.7 * N(t))
- Focus your analysis on the temporal pattern of measured activity
- The data reveals the underlying decay law through the detector's response

**Physics Background:**
- The radiation detector measures decay events over time
- Detector efficiency affects the measured activity but preserves the temporal pattern
- The measured activity reveals the underlying decay law through the detector's response

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to decay behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., initial conditions vs. decay rates)
- **Note**: The law in this universe may be different from the law in the real world. You must output the ground truth law of the isotope decay.

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Detector Efficiency: Radiation detector has 70% efficiency
   - Only 70% of actual decay events are detected

3. Activity Measurement: Detector measures decay events over time
   - The detector counts decay events in one-second intervals
   - Measured activity reveals the underlying decay law
   - Efficiency factor affects magnitude but not temporal pattern"""

# Complex system discovery prompt (two-isotope radioactive decay)
COMPLEX_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a specially prepared radioactive decay system containing two different radioactive isotopes (Isotope A and Isotope B) that coexist in the initial sample. The system can:
1. Control initial populations and lambda constants for both isotopes
2. Track the ratio of isotope populations over time
3. Measure the instantaneous ratio R(t) = Nₐ(t) / Nᵦ(t) using an advanced spectrometer
4. Calculate ratio patterns from temporal data

**Control Parameters:**
- `N0a`: Initial number of nuclei for Isotope A (always positive)
- `N0b`: Initial number of nuclei for Isotope B (always positive)
- `lambda_a`: Lambda constant for Isotope A (always positive)
- `lambda_b`: Lambda constant for Isotope B (always positive)
- `t`: Time elapsed (always positive)
- `num_points`: Number of time points to sample

**Important Notes:**
- This is a complex two-isotope radioactive decay model
- The system provides temporal ratio profile data
- The observable is the ratio R(t) = Nₐ(t) / Nᵦ(t) between the two populations
- Isotopes are not in a decay chain - they simply coexist and decay independently
- If your input results in overflowing the system, the system will return `NaN` (Not a Number)

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"N0a": ..., "N0b": ..., "lambda_a": ..., "lambda_b": ..., "t": ..., "num_points": ...}},
   {{"N0a": ..., "N0b": ..., "lambda_a": ..., "lambda_b": ..., "t": ..., "num_points": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of temporal ratio profile data objects (at most 20 data points per experiment):
<experiment_output>
[
   {{"time": [...], "ratio": [...]}} ,
   {{"time": [...], "ratio": [...]}}
]
</experiment_output>

**Important Data Handling Note:**
- Some ratio values may appear as `NaN` (Not a Number) in the output
- **Ignore all NaN values** - they occur due to mathematical overflow or division by zero
- NaN values typically appear when one isotope population becomes extremely small
- Focus your analysis on the valid numerical ratio values
- The NaN values do not contain useful information for discovering the decay law

**Physics Background:**
- The ratio R(t) = Nₐ(t) / Nᵦ(t) reveals the relative decay behavior
- Different lambda constants lead to different rates for each isotope
- Ratio patterns reveal the underlying two-isotope decay law governing the system

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to decay behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., initial conditions vs. decay rates)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Independent Decay: Isotopes decay independently
   - Isotope A and B are not in a decay chain
   - They simply coexist and decay according to their own laws
   - The ratio R(t) reveals the relative decay dynamics

3. Ratio Calculation: Two-isotope decay equation methods
   - The system calculates ratios using: R(t) = Nₐ(t) / Nᵦ(t)
   - This provides discrete ratio values at each time point
   - The ratio pattern reveals the underlying two-isotope decay law governing the system

4. Ratio Profile Physics:
   - Ratios follow patterns based on relative decay behavior
   - The temporal pattern depends on both lambda constants and initial populations
   - This pattern reveals the underlying two-isotope decay law"""

# General disclaimer for this simulated universe
GENERAL_DISCLAIMER = """**IMPORTANT NOTE:** The physics laws in this simulated universe may differ from real-world physics. Focus on discovering the mathematical relationships that govern this specific simulation environment."""

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
def test_hypothesis(N0, lambda_constant, t):
    # Test your hypothesis: N(t) = N0 * e^(-lambda * t^2.5)
    import numpy as np
    return N0 * np.exp(-lambda_constant * (t ** 2.5))

# Test with different parameters
test_N0 = [1000.0, 2000.0, 3000.0]
test_lambda = [0.1, 0.2, 0.3]
test_t = [1.0, 2.0, 3.0]

for N0_val, lambda_val, t_val in zip(test_N0, test_lambda, test_t):
    remaining = test_hypothesis(N0_val, lambda_val, t_val)
    print(f"N0={N0_val}, λ={lambda_val}, t={t_val} → N(t)={remaining}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
N0=1000.0, λ=0.1, t=1.0 → N(t)=1000.0
N0=2000.0, λ=0.2, t=2.0 → N(t)=2000.0
N0=3000.0, λ=0.3, t=3.0 → N(t)=3000.0

**Your Code:**
```python
def test_hypothesis(N0, lambda_constant, t):
    # Test your hypothesis: N(t) = N0 * e^(-lambda * t^2.5)
    import numpy as np
    return N0 * np.exp(-lambda_constant * (t ** 2.5))

# Test with different parameters
test_N0 = [1000.0, 2000.0, 3000.0]
test_lambda = [0.1, 0.2, 0.3]
test_t = [1.0, 2.0, 3.0]

for N0_val, lambda_val, t_val in zip(test_N0, test_lambda, test_t):
    remaining = test_hypothesis(N0_val, lambda_val, t_val)
    print(f"N0={N0_val}, λ={lambda_val}, t={t_val} → N(t)={remaining}")
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
