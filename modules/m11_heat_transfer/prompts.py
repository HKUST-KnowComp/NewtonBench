from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """- m: mass of the substance. It should be a positive real number.
- c: It should be a positive real number.
- delta_T: temperature change. It should be a positive real number."""

# Heat transfer-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(m, c, delta_T):"
RETURN_DESCRIPTION = "the heat transfer based on the ground truth heat transfer law"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(m, c, delta_T):
   return m * c * delta_T
</final_law>

**Example 2:**
<final_law>
def discovered_law(m, c, delta_T):
   import math
   return m * c * delta_T
</final_law>

**Note:** 
- m is the mass
- c is the c_constant
- delta_T is the temperature change
- The function returns the heat transfer"""

# Vanilla Equation Prompt for heat transfer
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a heat transfer measurement device that can measure the heat transfer between materials. You have precise control over the following properties for each experiment you run:
- Mass (`m`) - always positive
- C_constant (`c`) - always positive
- Temperature Change (`delta_T`) - can be positive

**Important Notes:**
- Mass `m` is always positive
- C_constant `c` is always positive
- Temperature change `delta_T` must be positive (heating only)
- The heat transfer represents the amount of energy transferred due to temperature change

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat transfer behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls material properties, another controls driving force)

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names
- `delta_T` MUST be positive (greater than 0)
- The system only models heating processes
- If you need to model cooling, use the absolute value of the temperature decrease
- Negative delta_T values will cause the experiment to fail

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

*Your Request:*
<run_experiment>
[
  {{"m": ..., "c": ..., "delta_T": ...}},
  {{"m": ..., "c": ..., "delta_T": ...}}
]
</run_experiment>

*System Response:*
The system will return the a list of measured heat transfernt.
<experiment_output>
[1.234e+03, 2.345e+04]
</experiment_output>

**Note:** The system returns heat transfer values in Joules."""

# Simple system discovery prompt
SIMPLE_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a heat transfer system that can:
1. Control mass, c_constant, temperature change, and time parameters
2. Track heat transfer profiles over time
3. Measure heat transfer patterns

**Control Parameters:**
- `m`: Mass (always positive)
- `c`: c_constant (always positive)
- `delta_T`: Temperature change (MUST be positive, heating only)
- `t`: Time elapsed (always positive)
- `num_points`: Number of time points to sample

**CRITICAL: Temperature Change Constraint**
- `delta_T` MUST be positive (greater than 0)
- The system only models heating processes
- If you need to model cooling, use the absolute value of the temperature decrease
- Negative delta_T values will cause the experiment to fail

**Important Notes:**
- This is a simplified heat transfer model with power distribution among mechanisms
- The system calculates total heat transfer, converts it to power, applies energy loss, and distributes remaining power
- Heat transfer follows the ground truth law based on mass, c_constant, and temperature change

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"m": ..., "c": ..., "delta_T": ...}},
   {{"m": ..., "c": ..., "delta_T": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return a list of power distribution data objects:
<experiment_output>
[
   {{"P_cond": ..., "P_conv": ..., "P_rad": ...}} ,
   {{"P_cond": ..., "P_conv": ..., "P_rad": ...}}
]
</experiment_output>

**Physics Background:**
- Heat transfer follows the ground truth law: Q_total = ground_truth_law(m, c, delta_T)
- Characteristic time is calculated as: t = (m * c) / 100
- We then calculate the total power
- **Energy Loss Before Distribution**: 18-22% of energy is lost before distribution
- The remaining of power is distributed among three mechanisms:
  * Conduction (P_cond): Direct molecular transfer power through material
  * Convection (P_conv): Heat transfer power through fluid motion
  * Radiation (P_rad): Electromagnetic heat transfer power
- **CRITICAL**: The system models heating processes only (delta_T > 0)
- Temperature change represents the increase in temperature during heating
- **IMPORTANT**: For cooling processes, use the absolute value of temperature decrease
- **WARNING**: Negative delta_T values will cause experiment failure

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat transfer behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls material properties, another controls driving force)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Heat Transfer Law: Heat transfer follows the ground truth law
   - Q_total = ground_truth_law(m, c, delta_T) where Q_total is the total heat transfer
   - Heat transfer occurs at specific values based on mass, c_constant, and temperature change
   - The total heat transfer is converted to power using characteristic time
   - Temperature change must be positive (heating processes only)

3. Power Distribution Physics:
   - Total heat transfer follows the ground truth law based on mass, c_constant, and temperature change
   - The total heat transfer is converted to power using characteristic time: t = (m * c) / 100
   - **Energy Loss**: 18-22% of power is lost before distribution due to material imperfections
   - Your job is to discover the ground truth law for the total heat transfer Q_total

4. Temperature Constraint:
   - **delta_T MUST be positive (delta_T > 0)**
   - **Heating processes only** - the system cannot handle negative temperature changes
   - **For cooling processes**: Use the absolute value of temperature decrease
   - **Example**: If temperature decreases by 10K, use delta_T = 10 (not -10)
   - **Failure condition**: Negative delta_T will cause experiment failure"""

# Complex system discovery prompt
COMPLEX_SYSTEM_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a heat transfer system that can:
1. Control material properties, temperature changes, and power generation
2. Calculate heat transfer differences between parameter variations
3. Convert heat transfer differences to electrical power
4. Power light bulbs based on available power from heat transfer

**Control Parameters:**
- `m`: Mass (always positive)
- `c`: c_constant (always positive)
- `delta_T`: Temperature change (MUST be positive, heating only)

**CRITICAL: Temperature Change Constraints**
- `delta_T` MUST be positive (greater than 0)
- The system only models heating processes
- If you need to model cooling, use the absolute value of the temperature decrease
- Negative delta_T values will cause the experiment to fail

**Important Notes:**
- This is a difficult heat transfer model using light bulb power system
- The system calculates heat transfer differences between parameter variations
- Power is generated from heat transfer differences and used to light bulbs
- Each light bulb requires 1 Watt of power
- Your job is to discover the ground truth law: Q_original = ground_truth_law(m, c, delta_T)

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below
- If your format is incorrect, the system will ask you to read the initial prompt again
- Double-check your JSON syntax before submitting
- Ensure all required parameters are included with correct names

**Input/Output Format:**
<run_experiment>
[
   {{"m": ..., "c": ..., "delta_T": ...}},
   {{"m": ..., "c": ..., "delta_T": ...}}
]
</run_experiment>

**REMINDER: If you make a format error, re-read the initial prompt carefully to understand the correct format.**

**System Response:**
The system will return the number of light bulbs that can be powered:
<experiment_output>
[
   <number_of_light_bulbs>,
   <number_of_light_bulbs>
]
</experiment_output>

**Note**: Each light bulb requires 1 Watt. The count represents how many light bulbs can be powered from the heat transfer difference.

**Physics Background:**
- Heat transfer follows the ground truth law: Q = ground_truth_law(m, c, delta_T)
- The system calculates heat transfer for your input parameters (m, c, delta_T)
- It then calculates heat transfer for modified parameters (m, c_alt, delta_T) where c_alt is randomly generated
- The difference in heat transfer represents available energy for power generation
- Power is calculated as: P = Q_difference / t where t = (m * c) / 100
- Energy loss (18-22%) is applied before power calculation
- Each light bulb requires 1 Watt, so the output is the number of light bulbs that can be powered
- **CRITICAL**: The system models heating processes only (delta_T > 0)
- Temperature change represents the increase in temperature during heating
- **WARNING**: Negative delta_T values will cause experiment failure

**Strategy**: Analyze whether these parameters serve similar or different functions:
- **Similar roles**: Parameters that both contribute to heat transfer behavior in the same way
- **Different roles**: Parameters that control fundamentally different aspects (e.g., one controls material properties, another controls driving force)

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Conservation of Energy: Energy is conserved in the system
   - Energy cannot be created or destroyed, only transferred
   - Total energy in the system remains constant

2. Heat Transfer Difference Law: Power comes from heat transfer differences
   - Q_original = ground_truth_law(m, c, delta_T)
   - Q_alternative = ground_truth_law(m, c_alt, delta_T), where c_alt is randomly generated 
   - Q_difference = |Q_original - Q_alternative|
   - Power = Q_difference / t where t = (m * c) / 100

3. Recovery Path: How to discover the ground truth law
   - Use function mode to get direct Q_total measurements
   - Compare with light bulb counts to understand the difference mechanism
   - Analyze how light bulb count varies with m, c, delta_T
   - The ground truth law can be recovered by understanding the relationship between Q_original and Q_alternative

4. Energy Loss and Power Generation:
   - Energy loss (18-22%) is applied before power calculation
   - Power generation depends on heat transfer differences, not absolute values
   - Light bulb count provides indirect measurement of heat transfer relationships"""

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
def test_hypothesis(m, c, delta_T):
    return m * c * delta_T

# Test with different parameters
test_m = [1.0, 2.0, 3.0]
test_c = [1000.0, 2000.0, 3000.0]
test_delta_T = [5.0, 10.0, 15.0]

for m_val, c_val, delta_T_val in zip(test_m, test_c, test_delta_T):
    heat_transfer = test_hypothesis(m_val, c_val, delta_T_val)
    print(f"m={m_val}, c={c_val}, delta_T={delta_T_val} → Q={heat_transfer}")
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
m=1.0, c=1000.0, delta_T=5.0 → Q=5000.0
m=2.0, c=2000.0, delta_T=10.0 → Q=40000.0
m=3.0, c=3000.0, delta_T=15.0 → Q=135000.0

**Your Code:**
```python
def test_hypothesis(m, c, delta_T):
    return m * c * delta_T

# Test with different parameters
test_m = [1.0, 2.0, 3.0]
test_c = [1000.0, 2000.0, 3000.0]
test_delta_T = [5.0, 10.0, 15.0]

for m_val, c_val, delta_T_val in zip(test_m, test_c, test_delta_T):
    heat_transfer = test_hypothesis(m_val, c_val, delta_T_val)
    print(f"m={m_val}, c={c_val}, delta_T={delta_T_val} → Q={heat_transfer}")
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
