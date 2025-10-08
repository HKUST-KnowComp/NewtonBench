from .m1_types import ExperimentSystem
from modules.common.prompts_base import (
	OBJECTIVE_PROMPT,
	ASSISTING_LAWS_DISCLAIMER,
	SUBMISSION_REQUIREMENTS,
	RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
	RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- q1: charge of the first object. It can be any real number.
- q2: charge of the second object. It can be any real number.
- distance: distance between the two objects. It should be a positive real number."""

# Electrostatics-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(q1, q2, distance):"
RETURN_DESCRIPTION = "the signed magnitude of the electrostatic force between the two charges (positive means repulsive; negative means attractive)"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(q1, q2, distance):
   import math
   C = 9e9
   return (C * q1 * q2) / (math.pow(distance, 2))
</final_law>

**Example 2:**
<final_law>
def discovered_law(q1, q2, distance):
   C = 9e9
   return (C * q1 * q2) / (distance ** 2)
</final_law>

**Note:** 
- q1 and q2 can be negative (electrons) or positive (protons)
- distance is always positive, so no abs() needed"""

# Vanilla equation discovery prompt for electrostatics
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a device that can position two point charges and measure the force acting between them. You have precise control over the following properties for each experiment you run:
- Charge of the first object (`q1`) - can be positive or negative
- Charge of the second object (`q2`) - can be positive or negative  
- Distance between the charges (`distance`) - always positive

**Important Notes:**
- Charges `q1` and `q2` can be negative (electrons) or positive (protons)
- The sign of `q1` and `q2` are critical. Their product determines whether the force will be repulsive (positive) or attractive (negative). Therefore, do not use abs() on your final law.
- Distance `distance` is always positive and represents the magnitude of separation
- No need to use abs() on distance since it's guaranteed to be positive

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"q1": ..., "q2": ..., "distance": ...}},
  {{"q1": ..., "q2": ..., "distance": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured force.
<experiment_output>
[1.234e-05, 2.345e-04]
</experiment_output>"""

COULOMB_1D_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a 1D motion tracking system that can:
1. Position two charges (q1, q2) at any 1D coordinates
2. Both charges decay exponentially over time with a fixed decay rate: q(t) = q(0) * exp(-t/decay_rate)
3. Decay rate will not be disclosed to you
4. Track the electric field at each charge over time

**Control Parameters:**
- `q1`: Initial charge of the fixed object at origin (can be positive or negative)
- `m1`: Mass of q1 (use np.inf to ensure it's fixed)
- `q2`: Initial charge of the moving object (can be positive or negative)
- `m2`: Mass of q2 (affects acceleration via F=ma)
- `distance`: Initial distance between charges (always positive)
- `duration`: Time to track motion
- `time_step`: Time interval between measurements

**Important Notes:**
- Charges `q1` and `q2` can be negative (electrons) or positive (protons)
- The sign of `q1` and `q2` are critical. Their product determines whether the force will be repulsive (positive) or attractive (negative). Therefore, do not use abs() on your final law.
- Distance `distance` is always positive and represents the initial separation
- Masses `m1` and `m2` affect the acceleration via F=ma
- The system provides time and velocity data over time

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"q1": ..., "m1": ..., "q2": ..., "m2": ..., "distance": ..., "duration": ..., "time_step": ...}},
   {{"q1": ..., "m1": ..., "q2": ..., "m2": ..., "distance": ..., "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "velocity": [...]}} ,
   {{"time": [...], "velocity": [...]}}
]
</experiment_output>

**Physics Background:**
- Motion is restricted to one dimension along the x-axis
- Both q1 and q2 can move according to F=ma dynamics

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Newton's Second Law: F = ma
   - F is the force
   - m is the mass
   - a is the acceleration

2. Kinematic Equations:
   - v = v₀ + a * t 
   - x = x₀ + v * t 
"""

COULOMB_1D_DISCOVERY_PROMPT_WITH_KINETIC_ENERGY = """**Experimental Apparatus:**

You have access to a 1D motion tracking system that can:
1. Position two charges (q1, q2) at any 1D coordinates
2. Both charges decay exponentially over time with a fixed decay rate: q(t) = q(0) * exp(-t/decay_rate)
3. Decay rate will not be disclosed to you
4. Track the electric field at each charge over time
5. Track q2's kinetic energy as it moves along the x-axis

**Control Parameters:**
- `q1`: Initial charge of the fixed object at origin (can be positive or negative)
- `m1`: Mass of q1 (use np.inf to ensure it's fixed)
- `q2`: Initial charge of the moving object (can be positive or negative)
- `m2`: Mass of q2 (affects acceleration via F=ma)
- `distance`: Initial distance between charges (always positive)
- `duration`: Time to track motion
- `time_step`: Time interval between measurements

**Important Notes:**
- Charges `q1` and `q2` can be negative (electrons) or positive (protons)
- The sign of `q1` and `q2` are critical. Their product determines whether the force will be repulsive (positive) or attractive (negative). Therefore, do not use abs() on your final law.
- Distance `distance` is always positive and represents the initial separation
- Masses `m1` and `m2` affect the acceleration via F=ma
- The system provides time and kinetic energy data over time

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"q1": ..., "m1": ..., "q2": ..., "m2": ..., "distance": ..., "duration": ..., "time_step": ...}},
   {{"q1": ..., "m1": ..., "q2": ..., "m2": ..., "distance": ..., "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "kinetic_energy": [...]}} ,
   {{"time": [...], "kinetic_energy": [...]}}
]
</experiment_output>

**Physics Background:**
- Motion is restricted to one dimension along the x-axis
- Both q1 and q2 can move according to F=ma dynamics

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Newton's Second Law: F = ma
   - F is the force
   - m is the mass
   - a is the acceleration

2. Kinematic Equations:
   - v = v₀ + a * t 
   - x = x₀ + v * t 

3. Kinetic Energy: KE = ½mv²
"""

COULOMB_2D_DISCOVERY_PROMPT = """**Experimental Apparatus:**

You have access to a 2D field simulation system that can:
1. Position two charges (q1, q2) at any 2D coordinates
2. Both charges decay exponentially over time with a fixed decay rate: q(t) = q(0) * exp(-t/decay_rate)
3. Decay rate will not be disclosed to you
4. Track the electric field vectors E1 and E2 at each charge position over time

**Control Parameters:**
- `q1`: Initial charge of the first particle (can be positive or negative)
- `pos1`: The [x, y] position of the first particle
- `q2`: Initial charge of the second particle (can be positive or negative)
- `pos2`: The [x, y] position of the second particle
- `duration`: Time to track the electric fields
- `time_step`: Time interval between measurements

**Important Notes:**
- Charges `q1` and `q2` can be negative (electrons) or positive (protons)
- The sign of `q1` and `q2` are critical. Their product determines whether the force will be repulsive (positive) or attractive (negative). Therefore, do not use abs() on your final law.
- Distance calculations between charges will always be positive
- No need to use abs() on distance since it's guaranteed to be positive

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"q1": ..., "pos1": [x1, y1], "q2": ..., "pos2": [x2, y2], "duration": ..., "time_step": ...}},
   {{"q1": ..., "pos1": [x1, y1], "q2": ..., "pos2": [x2, y2], "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "E1_x": [...], "E1_y": [...], "E2_x": [...], "E2_y": [...]}} ,
   {{"time": [...], "E1_x": [...], "E1_y": [...], "E2_x": [...], "E2_y": [...]}}
]
</experiment_output>

**Physics Background:**
- The simulation tracks electric field vectors in 2D space
- E1_x, E1_y are the x and y components of the electric field at position 1
- E2_x, E2_y are the x and y components of the electric field at position 2
- The electric field is calculated as E = F/q where F is the force and q is the charge

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Coulomb's Law: F = qE
   - F is the force vector
   - q is the charge
   - E is the electric field vector

2. Decay Law: q(t) = q(0) * exp(-t/50)
   - q(0) is the initial charge
   - t is the time

3. Superposition Principle: The net electric field is the vector sum of fields from all charges"""

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
def test_hypothesis(q1, q2, distance):
	# Test your hypothesis: F = K * q1 * q2 / r^3
	K = 2.0
	return (K * q1 * q2) / (distance ** 3)

# Test with different parameters
test_q1 = [1.0, -1.0, 2.0]
test_q2 = [1.0, 1.0, -1.0]
test_distance = [1.0, 2.0, 3.0]

for q1_val, q2_val, d_val in zip(test_q1, test_q2, test_distance):
	force = test_hypothesis(q1_val, q2_val, d_val)
	print(f"q1={q1_val}, q2={q2_val}, d={d_val} → F={force}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
q1=1.0, q2=1.0, d=1.0 → F=2.0
q1=-1.0, q2=1.0, d=2.0 → F=-0.25
q1=2.0, q2=-1.0, d=3.0 → F=-0.0741

**Your Code:**
```python
def test_hypothesis(q1, q2, distance):
	# Test your hypothesis: F = K * q1 * q2 / r^3
	K = 2.0
	return (K * q1 * q2) / (distance ** 3)

# Test with different parameters
test_q1 = [1.0, -1.0, 2.0]
test_q2 = [1.0, 1.0, -1.0]
test_distance = [1.0, 2.0, 3.0]

for q1_val, q2_val, d_val in zip(test_q1, test_q2, test_distance):
	force = test_hypothesis(q1_val, q2_val, d_val)
	print(f"q1={q1_val}, q2={q2_val}, d={d_val} → F={force}")
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
		prompts.append(COULOMB_1D_DISCOVERY_PROMPT.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))
	elif system == ExperimentSystem.COMPLEX_SYSTEM:
		prompts.append(COULOMB_1D_DISCOVERY_PROMPT_WITH_KINETIC_ENERGY.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))

	# Add code assisted instructions if requested
	if is_code_assisted:
		prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
	
	prompts.append(SUBMISSION_REQUIREMENTS.format(
		function_signature=FUNCTION_SIGNATURE,
		return_description=RETURN_DESCRIPTION,
		example=EXAMPLE
	))
	return "\n\n".join(prompts)
