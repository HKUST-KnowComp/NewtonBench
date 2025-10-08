from .m6_types import ExperimentSystem
from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- k: spring constant. It should be a positive real number.
- m: mass of the object. It should be a positive real number.
- b: damping coefficient. It should be a positive real number."""

# --- Core Law Discovery ---
FUNCTION_SIGNATURE = "def discovered_law(k, m, b):"
RETURN_DESCRIPTION = "the angular velocity (ω) of the damped oscillator"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(k, m, b):
    import math
    return math.sqrt(k/m - (b/(2*m))**2)
</final_law>

**Note**:
- k is the spring constant
- m is the mass
- b is the damping coefficient"""

VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a system to measure the angular velocity of a damped harmonic oscillator. You have precise control over the following properties for each experiment you run:
**Control Parameters:**
- `k`: Spring constant.
- `m`: Mass of the object.
- `b`: Damping coefficient.

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
    {{"k": ..., "m": ..., "b": ...}},
    {{"k": ..., "m": ..., "b": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measured angular velocity.
<experiment_output>
[..., ...]
</experiment_output>"""

# --- Simple System: The Damped Oscillator ---
SIMPLE_SYSTEM_PROMPT = """**Experimental Apparatus:**
You have a damped oscillator setup. You can control the physical properties of the system and measure the resulting period of oscillation.

**Experimental Setup:**
1. **Spring-Mass System**: A mass `m` is attached to a spring with constant `k`.
2. **Damping Medium**: The system is placed in a fluid that provides a damping force proportional to the velocity, with a damping coefficient `b`.
3. **Measurement**: The apparatus measures the time it takes for the system to complete one full oscillation (the period `T`).

**Control Parameters:**
- `spring_constant`: The spring constant (k).
- `mass`: The mass of the object (m).
- `damping_coefficient`: The damping coefficient (b).

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
  {{"spring_constant": ..., "mass": ..., "damping_coefficient": ...}},
  {{"spring_constant": ..., "mass": ..., "damping_coefficient": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the period:
<experiment_output>
[
  {{"period": "..."}},
  {{"period": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Period and Angular Velocity:** T = 2π / ω

2.  **Superposition of Forces:** 
    - The net force is the vector sum of all individual forces acting on the mass."""

# --- Complex System: The Damped Oscillator ---
COMPLEX_SYSTEM_PROMPT = """**Experimental Apparatus:**
You are working with a damped harmonic oscillator system. The setup consists of a mass attached to a spring and subject to a damping force. You can control the physical properties of the system and observe how the amplitude of oscillation evolves over time.

**Experimental Setup:**
1. **Spring-Mass System:**
    - A mass m is attached to a spring with spring constant k, forming a classic harmonic oscillator.
2. **Damping Medium:**
    - The system is immersed in a fluid that exerts a damping force proportional to the velocity of the mass. This damping is characterized by the coefficient b.
3. **Initial Conditions:**
    - The system is displaced with an initial amplitude A₀ and then released to oscillate freely.
4. **Measurement:**
    - The apparatus records the amplitude of oscillation over time. Data is collected over two periods, with a maximum of 20 time-amplitude points per experiment.

**Control Parameters:**
- `spring_constant`: The spring constant (k).
- `mass`: The mass of the object (m).
- `damping_coefficient`: The damping coefficient (b).
- `initial_amplitude`: The initial amplitude (A0).

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
  {{"spring_constant": ..., "mass": ..., "damping_coefficient": ..., "initial_amplitude": ...}},
  {{"spring_constant": ..., "mass": ..., "damping_coefficient": ..., "initial_amplitude": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the timestamp and the amplitude.
<experiment_output>
[
  {{"time": [...], "amplitude": [...]}},
  {{"time": [...], "amplitude": [...]}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Amplitude of a Damped Oscillator:** x(t) = A₀ * e^(-3*t) * cos(ω*t)

2.  **Period and Angular Velocity:** T = 2π / ω

3.  **Superposition of Forces:** 
    - The net force is the vector sum of all individual forces acting on the mass."""

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
def test_hypothesis(k, m, b):
    import math
    try:
        return math.sqrt(k/m - (b/(2*m))**2)
    except ValueError:
        return float('nan')

# Test with different parameters
print(test_hypothesis(10.0, 1.0, 0.5))
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
3.122498999199199

**Your Code:**
```python
def test_hypothesis(k, m, b):
    import math
    try:
        return math.sqrt(k/m - (b/(2*m))**2)
    except ValueError:
        return float('nan')

# Test with different parameters
print(test_hypothesis(10.0, 1.0, 0.5))
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
    
    # Add code assisted instructions if requested
    if is_code_assisted:
        prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
    
    prompts.append(SUBMISSION_REQUIREMENTS.format(
        function_signature=FUNCTION_SIGNATURE,
        return_description=RETURN_DESCRIPTION,
        example=EXAMPLE
    ))
    
    return "\n\n".join(prompts) 
