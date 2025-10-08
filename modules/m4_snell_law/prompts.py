from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    ASSISTING_LAWS_DISCLAIMER,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)
from .m4_types import ExperimentSystem

PARAM_DESCRIPTION = """- n1: refractive index of the first medium. It should be a positive real number, typically >= 1.
- n2: refractive index of the second medium. It should be a positive real number, typically >= 1.
- angle1: angle of incidence in degrees. It should be a real number, typically between 0 and 90."""

# --- Vanilla Equation Prompt ---
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a precision refractometer. You have precise control over the following properties for each experiment you run:
- The refractive index of the first medium (n₁) (ranging from 1 to 1.5)
- The refractive index of the second medium (n₂) (ranging from 1 to 1.5)
- The angle of incidence (θ₁) in degrees (ranging from 0 to 90)

**Strategy**: Analyze the functional roles of parameters in relation to the refraction angle:
- **Similar roles**: Identify parameters that influence the refraction angle in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system (e.g., one controls angle, another controls medium properties).
- **Change of subject**: Instead of analyzing the angle directly, explore transformations or derived relationships involving the angle. For example:
        - Apply trigonometric functions such as sin(θ₂), tan(θ₂), or cos(θ₂)
        - Investigate whether these transformed values reveal hidden patterns or simplify the law
- **Optically Denser to Less Dense Medium**:    
    - When light travels from an optically denser medium to a less dense medium, refraction may not occur under certain conditions, especially when the difference in refractive indices is large.
    - To ensure meaningful results during experimentation for refraction, consider using media with closer refractive indices.
- **Higher variety of input**:
    - To explore a wider range of refraction behaviors, avoid using n₁ = 1 or n₂ = 1 as default values. Instead, test with a variety of refractive indices (e.g., 1.1, 1.12, etc.).

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
  {{"refractive_index_1": ..., "refractive_index_2": ..., "incidence_angle": ...}},
  {{"refractive_index_1": ..., "refractive_index_2": ..., "incidence_angle": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measured refraction angle (θ₂).
<experiment_output>
[..., ...]
</experiment_output>"""

# --- Simple System: Light Propagation ---
LIGHT_PROPAGATION_PROMPT = """**Experimental Apparatus:**
You have access to a two-medium optical bench. A monochromatic light beam travels from Medium 1 to Medium 2. The apparatus measures the refraction angle in Medium 2 (θ₂).

**Experimental Setup:**
1. Medium 1: You control the speed of light (v₁)
2. Medium 2: You control the speed of light (v₂)
3. Beam: You control the incidence angle (θ₁)
4. Measurement: The apparatus returns the refraction angle (θ₂)

**Controllable Parameters:**
- `speed_medium1`: Speed of light in Medium 1 (must be > 0 and <= 3.0e8 m/s)
- `speed_medium2`: Speed of light in Medium 2 (must be > 0 and <= 3.0e8 m/s)
- `incidence_angle`: Angle of incidence θ₁ in degrees (ranging from 0 to 90)

**Strategy**: Analyze the functional roles of parameters in relation to the refraction angle:
- **Similar roles**: Identify parameters that influence the refraction angle in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system (e.g., one controls angle, another controls medium properties).
- **Change of subject**: Instead of analyzing the angle directly, explore transformations or derived relationships involving the angle. For example:
        - Apply trigonometric functions such as sin(θ₂), tan(θ₂), or cos(θ₂)
        - Investigate whether these transformed values reveal hidden patterns or simplify the law
- **Optically Denser to Less Dense Medium**:    
    - When light travels from an optically denser medium to a less dense medium, refraction may not occur under certain conditions, especially when the difference in refractive indices is large.
    - To ensure meaningful results during experimentation for refraction, consider using media with closer refractive indices.

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
  {{"speed_medium1": ..., "speed_medium2": ..., "incidence_angle": ...}},
  {{"speed_medium1": ..., "speed_medium2": ..., "incidence_angle": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of measurement objects, each containing the refraction angle:
<experiment_output>
[
  {{"refraction_angle": "..."}},
  {{"refraction_angle": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Speed of Light Relationship: v = c / n
    - v is the speed of light in the medium; c is the speed of light in vacuum (3.0e8); n is the refractive index

2. Light Propagation: 
    - Light travels in straight lines within uniform media"""

# --- Complex System: Triple-Layer Refraction ---
TRIPLE_LAYER_PROMPT = """**Experimental Apparatus:**
You use a "Triple-Layer Refractometer." A light beam passes through a stack of three media. The apparatus only reports the initial angle of incidence and the final angle of refraction.

**Experimental Setup:**
1. **Layer 1**: First medium with known refractive index (n₁)  (ranging from 1 to 1.5)
2. **Layer 2**: Hidden middle layer with refractive index (n₂)  (ranging from 1 to 1.5)
3. **Layer 3**: Third medium with known refractive index (n₃)  (ranging from 1 to 1.5)
4. **Light Beam**: Enters at angle θ₁, exits at angle θ_final
5. **Hidden Information**: The apparatus only shows θ_final, not the intermediate angles

**Controllable Parameters:**
- `refractive_index_1`: Refractive index of the first layer (n₁)
- `refractive_index_2`: Refractive index of the hidden middle layer (n₂)
- `refractive_index_3`: Refractive index of the third layer (n₃)
- `incidence_angle`: Initial angle of incidence (θ₁) in degrees

**Strategy**: Analyze the functional roles of parameters in relation to the refraction angle:
- **Similar roles**: Identify parameters that influence the refraction angle in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system (e.g., one controls angle, another controls medium properties).
- **Change of subject**: Instead of analyzing the angle directly, explore transformations or derived relationships involving the angle. For example:
        - Apply trigonometric functions such as sin(θ₂), tan(θ₂), or cos(θ₂)
        - Investigate whether these transformed values reveal hidden patterns or simplify the law
- **Optically Denser to Less Dense Medium**:    
    - When light travels from an optically denser medium to a less dense medium, refraction may not occur under certain conditions, especially when the difference in refractive indices is large.
    - To ensure meaningful results during experimentation for refraction, consider using media with closer refractive indices.

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
  {{"refractive_index_1": ..., "refractive_index_2": ..., "refractive_index_3": ..., "incidence_angle": ...}},
  {{"refractive_index_1": ..., "refractive_index_2": ..., "refractive_index_3": ..., "incidence_angle": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of measurement objects, each containing the final refraction angle:
<experiment_output>
[
  {{"final_refraction_angle": "..."}},
  {{"final_refraction_angle": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. **Sequential Refraction:** 
    - Light refracts at each interface according to a shifted Snell's Law, which may differ from the classical formulation.
    - The angle of refraction at the interface between medium 1 and medium 2 becomes the angle of incidence at the next interface between medium 2 and medium 3.
    - This sequential behavior continues across all interfaces, forming a chain of refractions where each output angle becomes the input for the next."""

# --- Function Signature and Return Description ---
FUNCTION_SIGNATURE = "def discovered_law(n1, n2, angle1):"
RETURN_DESCRIPTION = "Return the refraction angle (angle2) in degrees"
EXAMPLE = """Example:
<final_law>
def discovered_law(n1, n2, angle1):
    import math
    import numpy as np
    try:
        return math.degrees(math.acos(n1 * math.sin(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')
</final_law>
    
**Note:** 
- n1 is the refractive index of the first medium
- n2 is the refractive index of the second medium
- angle1 is the incidence angle in degree"""

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
def test_hypothesis(n1, n2, angle1):
    import math
    try:
        return math.degrees(math.acos(n1 * math.sin(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

# Test with different parameters
print(test_hypothesis(1.0, 1.5, 30.0))
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
19.47122063449069

**Your Code:**
```python
def test_hypothesis(n1, n2, angle1):
    import math
    try:
        return math.degrees(math.acos(n1 * math.sin(math.radians(angle1)) / n2))
    except ValueError:
        return float('nan')

# Test with different parameters
print(test_hypothesis(1.0, 1.5, 30.0))
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
        prompts.append(LIGHT_PROPAGATION_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction, ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER))
    elif system == ExperimentSystem.COMPLEX_SYSTEM:
        prompts.append(TRIPLE_LAYER_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction, ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER))
    
    # Add code assisted instructions if requested
    if is_code_assisted:
        prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
    
    prompts.append(SUBMISSION_REQUIREMENTS.format(
        function_signature=FUNCTION_SIGNATURE,
        return_description=RETURN_DESCRIPTION,
        example=EXAMPLE
    ))
    
    return "\n\n".join(prompts) 