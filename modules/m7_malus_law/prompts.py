"""
LLM prompts and instructions for Module 7: Malus's Law

This module contains the prompts and instructions for the LLM to discover
the relationship between light intensity and polarization angle.
"""

from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """- I_0: initial intensity of the light. It should be a positive real number.
- theta: angle between the polarization axis of the polarizer and the polarization direction of the incident light in radians. It should be a real number between 0 and π/2."""

# Malus's Law-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(I_0, theta):"
RETURN_DESCRIPTION = "the mathematical formula for the ground truth Malus's Law that governs light transmission through polarizers in this universe"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(I_0, theta):
   import math
   return I_0 * (math.cos(theta) ** 2)
</final_law>

**Example 2:**
<final_law>
def discovered_law(I_0, theta):
   return I_0 * (math.cos(theta) ** 2)
</final_law>

**Note:** 
- I_0 is the initial light intensity (always positive)
- theta is the angle between polarization direction and polarizer axis (greater than 0 to π/2 radians)
- The function returns the transmitted light intensity"""

# Vanilla equation prompt for Malus's Law
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a polarized light transmission system that can measure how light intensity changes when passing through polarizers. You have precise control over the following properties for each experiment you run:
- Initial Intensity (`I_0`) - always positive
- Angle (`theta`) - between 0 and π/2 radians

**Important Notes:**
- Initial Intensity `I_0` is always positive and represents the starting light intensity
- Angle `theta` is always greater than 0 and less than or equal to π/2 radians (greater than 0° to 90°)
- The transmitted intensity represents how much light gets through the polarizer
- The system only models positive light intensities

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to transmission behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls initial conditions, another controls transmission efficiency)

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
  {{"I_0": ..., "theta": ...}},
  {{"I_0": ..., "theta": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured transmitted intensity.
<experiment_output>
[1.234e+02, 2.345e+02]
</experiment_output>"""

# Simple system prompt for Malus's Law
SIMPLE_SYSTEM_PROMPT = """**Simple Model Malus's Law Discovery Experiment**

You have access to a polarized light system that can:
1. Control initial light intensity and polarization angle
2. Calculate transmitted intensity using the ground truth law
3. Output intensity ratio data for analysis

**Control Parameters:**
- `I_0`: Initial light intensity in W/m² (always positive)
- `theta`: Angle between polarization direction and polarizer axis in radians. It must be greater than 0 and less than or equal to π/2 radians

**Important Notes:**
- This is a simple polarization transmission model
- The system calculates transmitted intensity using the ground truth law of the specified difficulty
- The observable is the intensity ratio I/I_0 (dimensionless)
- All ratio values are positive and represent the transmission efficiency
- Angle `theta` is always greater than 0 and less than or equal to π/2 radians 

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names
- Ensure that the angle `theta` is always greater than 0 and less than or equal to π/2 radians (greater than 0° to 90°)

**Input/Output Format:**
<run_experiment>
[
   {{"I_0": ..., "theta": ...}},
   {{"I_0": ..., "theta": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of intensity ratio values (one ratio per experiment):
<experiment_output>
[1.234e+00, 2.345e+00]
</experiment_output>

**Important Data Handling Note:**
- Each ratio value represents the transmission efficiency (I/I_0) for the given input parameters
- All ratio values are positive and dimensionless
- The ratio values reveal the underlying Malus's Law through their relationship to the input angle

**Physics Background:**
- Polarized light has a specific electric field direction
- Polarizers only transmit light aligned with their axis
- The transmission efficiency depends on the angle between polarization and polarizer
- The intensity ratio reveals the underlying transmission law

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to transmission behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., initial conditions vs. transmission efficiency)
- **Note**: The law in this universe may be different from the law in the real world. You must output the ground truth law of light transmission.

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Light energy cannot be created or destroyed, only transmitted or absorbed
   - Total energy in the system remains constant

2. Polarization Physics: Light polarization affects transmission
   - Polarized light has a specific electric field direction
   - Polarizers only transmit light aligned with their axis
   - The transmission depends on the angle between polarization and polarizer

3. Intensity Relationship: There is a mathematical relationship
   - Intensity ratio depends on the angle between polarization and polarizer
   - The relationship follows a specific mathematical law
   - This law can be discovered through systematic analysis of ratio patterns"""

# Complex system prompt for Malus's Law
COMPLEX_SYSTEM_PROMPT = """**Difficult Model Two-Polarizer System Discovery Experiment**

You have access to a complex three-polarizer system that can:
1. Control initial light intensity and polarization angle
2. Track intensity changes through two sequential polarizers
3. Measure the intensity difference after complex interactions
4. Calculate intensity difference patterns from angle series data

**Control Parameters:**
- `I_0`: Initial light intensity in W/m² (always positive)
- `theta`: Angle between polarization direction and polarizer axis in radians

**Important Notes:**
- This is a three-polarizer system: Polarizer 1 → Polarizer 2 → Polarizer 3
- The system calculates intensity difference through sequential polarizer interactions
- The observable is the intensity difference (I_1 - I_0) after two polarizer stages
- The system models realistic polarization physics with cumulative effects
- Angle `theta` is always greater than 0 and less than or equal to π/2 radians 

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names
- Angle `theta` is always greater than 0 and less than or equal to π/2 radians 

**Input/Output Format:**
<run_experiment>
[
   {{"I_0": ..., "theta": ...}},
   {{"I_0": ..., "theta": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of intensity difference values (one difference per experiment):
<experiment_output>
[1.234e+02, -2.345e+01]
</experiment_output>

**Important Data Handling Note:**
- Each intensity difference value shows the cumulative effect of two polarizers
- Intensity difference values can be positive or negative (I_1 - I_0)
- Focus your analysis on the relationship between input parameters and intensity difference
- The data reveals the underlying transmission law through cumulative polarizer effects

**Physics Background:**
- The system models a three-polarizer setup: Polarizer 1 → Polarizer 2 → Polarizer 3
- Light passes through two sequential polarizers, each applying the same transmission law
- The intensity difference (I_1 - I_0) reveals the cumulative effect of both polarizers
- The intensity difference reveals the underlying transmission law through sequential application

**Strategy**: Analyze the intensity difference values to discover the underlying law:
- **Cumulative Analysis**: The intensity difference (I_1 - I_0) reveals the effect of sequential polarizer application
- **Pattern Recognition**: Look for mathematical patterns in how the difference varies with input parameters
- **Law Discovery**: The difference values reveal the underlying transmission law through cumulative effects

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Light energy cannot be created or destroyed, only transmitted or absorbed
   - Total energy in the system remains constant

2. Sequential Polarization Physics: Two-polarizer system affects transmission
   - The system involves two sequential polarizers (Polarizer 1 → Polarizer 2 → Polarizer 3)
   - Each polarizer applies the same transmission law
   - The cumulative effect reveals the underlying transmission law

3. Intensity Difference Relationship: There is a mathematical relationship
   - Intensity difference (I_1 - I_0) depends on initial intensity and angle
   - The relationship follows a specific mathematical law
   - This law can be discovered through systematic analysis of difference values

4. Cumulative Transmission: Sequential polarizer effects
   - The system models realistic polarization physics with cumulative effects
   - Sequential polarizer interactions create intensity differences
   - The difference values reveal the underlying transmission law through cumulative application"""

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

# Example involving cosines and powers
theta = 0.785  # π/4 radians
cos_val = math.cos(theta)
power_val = cos_val ** 2
result = cos_val * power_val
print(f"cos({theta}) * cos²({theta}) = {result}")

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
def test_hypothesis(I_0, theta):
    # Test your hypothesis: I = I_0 * cos²(theta)
    import math
    return I_0 * (math.cos(theta) ** 2)

# Test with different parameters
test_I_0 = [1000.0, 2000.0, 3000.0]
test_theta = [0.001, 0.785, 1.571]  # Very small angle, 45°, 90°

for I_0_val, theta_val in zip(test_I_0, test_theta):
    transmitted = test_hypothesis(I_0_val, theta_val)
    print(f"I_0={I_0_val}, θ={theta_val} → I={transmitted:.2f}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
I_0=1000.0, θ=0.001 → I=1000.00
I_0=2000.0, θ=0.785 → I=1000.00
I_0=3000.0, θ=1.571 → I=0.00

**Your Code:**
```python
def test_hypothesis(I_0, theta):
    # Test your hypothesis: I = I_0 * cos²(theta)
    import math
    return I_0 * (math.cos(theta) ** 2)

# Test with different parameters
test_I_0 = [1000.0, 2000.0, 3000.0]
test_theta = [0.001, 0.785, 1.571]  # Very small angle, 45°, 90°

for I_0_val, theta_val in zip(test_I_0, test_theta):
    transmitted = test_hypothesis(I_0_val, theta_val)
    print(f"I_0={I_0_val}, θ={theta_val} → I={transmitted:.2f}")
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
