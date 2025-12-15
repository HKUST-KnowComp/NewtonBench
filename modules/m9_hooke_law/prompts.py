from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """- x: displacement from the equilibrium position. It should be a positive real number."""

# Hooke's law-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(x):"
RETURN_DESCRIPTION = "the elastic potential energy stored in the spring based on the ground truth Hooke's law. Note: If the calculated energy would be negative, it will be clamped to 0."
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(x):
   import math
   c = 0.5
   return cx
</final_law>

**Example 2:**
<final_law>
def discovered_law(x):
   c = 0.8
   return cx
</final_law>

**Note:** 
- x is the displacement from equilibrium
- The function returns the elastic potential energy stored
- If the calculated energy would be negative, it will be clamped to 0"""

# Vanilla Equation Prompt for Hooke's law
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a Hooke's law measurement device that can measure the elastic potential energy stored in springs. You have precise control over the following properties for each experiment you run:
- Displacement (`x`) - always positive

**Important Notes:**
- Displacement `x` is always positive
- The energy represents the amount of elastic potential energy stored in the spring
- If the calculated energy would be negative, it will be clamped to 0

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to energy storage behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls displacement)

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
  {{"x": ...}},
  {{"x": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured energy.
<experiment_output>
[1.234e-03, 2.345e-02]
</experiment_output>

**Note:** The system returns energy values in Joules where energy is stored in the spring."""

# Simple system discovery prompt
SIMPLE_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a Hooke's law system that can:
1. Control displacement and mass parameters
2. Calculate net kinetic energy after air resistance
3. Measure energy loss due to air resistance effects
4. Track velocity and energy conservation relationships

**Control Parameters:**
- `x`: Displacement from equilibrium (always positive)
- `m`: Mass (always positive)

**Important Notes:**
- This is a complex Hooke's law model with air resistance
- The system calculates net kinetic energy after accounting for air resistance
- Spring energy follows the ground truth law based displacement
- If the calculated spring energy would be negative, it will be clamped to 0
- k_air = 0.2 is known and constant
- The output is net kinetic energy after accounting for air resistance effects

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"x": ..., "m": ...}},
   {{"x": ..., "m": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of net kinetic energy values for each experiment:
<experiment_output>
[1.234e+00, 2.345e+00]
</experiment_output>

**Physics Background:**
- Hooke's law energy follows the ground truth law: U = ground_truth_law(x)
- Energy values depend on displacement according to the ground truth law
- The system calculates maximum velocity and then accounts for air resistance effects
- k_air = 0.2 is known and affects the final energy output
- The output represents net kinetic energy after air resistance losses

- **Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to energy storage behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls displacement)
- **Remember that your job is to discover the ground truth law for the net kinetic energy that takes x as an input**

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Hooke's Law Energy: Spring energy follows the ground truth law
   - U = ground_truth_law(x) where the exact form depends on the difficulty level
   - Energy storage occurs at specific values based on displacement

3. Air Resistance Physics:
   - k_air = 0.2 is known and constant
   - Air resistance affects the final kinetic energy output
   - The system accounts for energy losses due to air resistance
   - The output represents net kinetic energy after air resistance effects

4. Energy Conservation with Air Resistance:
   - Elastic potential energy: U = ground_truth_law(x)
   - Kinetic energy: U = 1/2 * m * v_max²
   - Air resistance energy loss: U_loss = -k_air * x * v_max²
   - Net kinetic energy: Net_KE = U - U_loss"""

# Complex system discovery prompt
COMPLEX_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a Hooke's law system that can:
1. Control displacement and mass parameters
2. Calculate realistic maximum velocity from spring energy
3. Account for energy losses in velocity calculations

**Control Parameters:**
- `x`: Displacement from equilibrium (always positive)
- `m`: Mass (always positive)
**Important Notes:**
- This is a simplified Hooke's law model with exponential energy loss
- The system calculates realistic maximum velocity from spring energy
- Spring energy follows the ground truth law based on displacement
- If the calculated spring energy would be negative, it will be clamped to 0
- The velocity accounts for energy losses and realistic physics factors

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"x": ..., "m": ...}},
   {{"x": ..., "m": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of realistic maximum velocity values in m/s for each experiment:
<experiment_output>
[1.234e+00, 2.345e+00]
</experiment_output>

**Physics Background:**
- Hooke's law energy follows the ground truth law: U = ground_truth_law(x)
- Energy values depend on displacement according to the ground truth law
- The system calculates maximum velocity using energy conservation: U = 1/2 * m * v_max^2
- **Exponential Energy Loss**: Energy retention follows exp(-x / x_scale) where x_scale is a fixed system parameter
- Larger displacements experience exponential energy loss due to material fatigue and geometric effects
- The system uses a characteristic displacement scale that determines how quickly energy decays with displacement


**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to energy storage behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls displacement)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Hooke's Law Energy: Spring energy follows the ground truth law
   - U = ground_truth_law(x) where the exact form depends on the difficulty level
   - Energy storage occurs at specific values based on displacement

3. Energy-Velocity Relationship:
   - Energy conservation: U = 1/2 * m * v_max^2
   - Realistic velocity accounts for displacement-dependent friction
   - The output represents achievable maximum velocity in practice
   - Your job is to discover the ground truth law for the U that takes x as an input

4. Exponential Energy Loss:
   - Energy retention follows exponential decay: energy_retention = exp(-x / x_scale)
   - The displacement scale x_scale is a fixed system parameter (not user-controlled)
   - Represents material fatigue, geometric nonlinearities, and stress accumulation"""

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
def test_hypothesis(x):
    c = 0.5
    return c*x

# Test with different parameters
test_x = [0.5, 1.0, 1.5]

for x_val in test_x:
    energy = test_hypothesis(x_val)
    print(f"x={x_val} → U={energy}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
2

**Your Code:**
```python
a = 1 + 1
print(a)
```
</python_output>
```

**System Response for Function Example:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
x=0.5 → U=0.125
x=1.0 → U=0.5
x=1.5 → U=1.125

**Your Code:**
```python
def test_hypothesis(x):
    c = 0.5
    return c*x

# Test with different parameters
test_x = [0.5, 1.0, 1.5]

for x_val in test_x:
    energy = test_hypothesis(x_val)
    print(f"x={x_val} → U={energy}")
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
