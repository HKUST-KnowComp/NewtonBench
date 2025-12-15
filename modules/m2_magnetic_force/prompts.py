from .m2_types import ExperimentSystem
from modules.common.prompts_base import (
	OBJECTIVE_PROMPT,
	ASSISTING_LAWS_DISCLAIMER,
	SUBMISSION_REQUIREMENTS,
	RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
	RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- current1: current in the first wire. It can be assumed that it is always positive.
- current2: current in the second wire. It can be assumed that it is always positive.
- distance: distance between the two wires. It should be a positive real number."""

# --- Core Law Discovery ---
FUNCTION_SIGNATURE = "def discovered_law(current1, current2, distance):"
RETURN_DESCRIPTION = "the magnitude of the magnetic force per unit length exerted on wire 2 by wire 1"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(current1, current2, distance):
   C = 2.0e-7
   return (C * current1 * current2) / (distance)
</final_law>

**Note**:
- the sign for the experiment input only controls the direction of the current. You can assume that the current1 and current2 for the discovered law are the magnitudes of the currents (always be positive), so don't need to add abs()
- distance is also always positive"""

VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a device that can set up two long, parallel wires and measure the magnetic force per unit length exerted on wire 2 by wire 1. You have precise control over:
- `current1`: Electric current in the first wire.
- `current2`: Electric current in the second wire.
- `distance`: The perpendicular distance between the wires. (always positive)

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"current1": ..., "current2": ..., "distance": ...}},
  {{"current1": ..., "current2": ..., "distance": ...}},
]
</run_experiment>

*System Response:*
The system will return a list of the measured force.
<experiment_output>
[1.235e-05, 1.341e-06]
</experiment_output>"""

# --- Simple System: Linear Motion ---
LINEAR_MOTION_PROMPT = """**Experimental Apparatus:**
You have access to a 1D motion tracking system that can:
1. Fix one long wire (Wire 1) at the origin (x=0).
2. Place a second, parallel wire (Wire 2), with a specific mass per unit length, at an initial position on the x-axis.
3. Give Wire 2 an initial velocity along the x-axis.
4. Track the position and velocity of Wire 2 over time.

**Control Parameters:**
- `current1`: Current in the fixed wire (Wire 1). (positive means z-direction, negative means -z direction)
- `current2`: Current in the moving wire (Wire 2). (positive means z-direction, negative means -z direction)
- `mass_wire`: The mass per unit length of the moving wire.
- `distance`: The initial starting position of Wire 2 on the x-axis. (always positive)
- `initial_velocity`: The initial velocity of Wire 2.
- `duration`: The time to track the motion.
- `time_step`: The time interval between measurements.
**Note:**
- The sign for the current is used to provide a simple way to control the direction of the current. You can assume that the current1 and current2 for the discovered law are the magnitudes of the currents, which will always be positive.

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"current1": ..., "current2": ..., "mass_wire": ..., "distance": ..., "initial_velocity": ..., "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "position": [...], "velocity": [...]}}
]
</experiment_output>

**Physics Background:**
- Motion is restricted to one dimension (the x-axis).
- The force between the wires acts along the x-axis.

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Newton's Second Law:** F = ma
   -   Where F is the force per unit length and m is the mass per unit length.
   -   Since motion is 1D, all vectors are scalars along the x-axis.

2.  **Kinematics Equations:**
   -   v = v₀ + at
   -   x = x₀ + v₀t + ½at²
   -   v² = v₀² + 2a(x - x₀)
   -   These equations relate position, velocity, and acceleration.

3.  **Magnetic force direction**:
   - when the currents have the same direction, Wire 2 will be attracted to Wire 1
   - when the currents have opposite directions, Wire 2 will be repelled by Wire 1"""

# --- Complex System: Fixed Wire Experiment ---
COMPLEX_SYSTEM_PROMPT = """**Experimental Apparatus:**
You have access to a device that can set up two long, parallel wires and track the motion of one wire under the influence of magnetic forces. The experiment setup is:
1. **Wire 1**: Fixed position, carries alternating current (AC) with fixed frequency 50Hz
2. **Wire 2**: Movable wire, carries direct current (DC)
3. Give Wire 2 an initial velocity.
4. Track the position and velocity of Wire 2 over time.
5. **Experiment Duration**: Always exactly 1 period (0.02 seconds at 50Hz)
6. **Data Output**: Exactly 20 data points of position and velocity over time

**Controllable Parameters:**
- `current1`: AC amplitude for wire 1 - controls the strength of the alternating magnetic field
- `current2`: DC current for wire 2 - controls the constant current in the moving wire
- `mass_wire`: Mass of the moving wire
- `distance`: Initial distance between the wires (always positive)
- `initial_velocity`: Initial velocity of the moving wire
**Note:**
- The sign for current1 controls the AC phase (positive = sine wave, negative = inverted sine wave)
- The sign for current2 controls the DC direction (positive means z-direction, negative means -z direction)
- The sign for the current is used to provide a simple way to control the direction of the current. You can assume that the current1 and current2 for the discovered law are the magnitudes of the currents, which will always be positive.
- The experiment will return exactly 20 data points showing how the wire moves over time

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"current1": ..., "current2": ..., "mass_wire": ..., "distance": ..., "initial_velocity": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "position": [...], "velocity": [...]}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Newton's Second Law:** F = ma
   -   Where F is the force per unit length and m is the mass per unit length.

2.  **Kinematics Equations:**
   -   v = v₀ + at
   -   x = x₀ + v₀t + ½at²
   -   v² = v₀² + 2a(x - x₀)
   -   These equations relate position, velocity, and acceleration.

3.  **Magnetic force direction**:
   - when the currents have the same direction, Wire 2 will be attracted to Wire 1
   - when the currents have opposite directions, Wire 2 will be repelled by Wire 1
   
4. **Alternating Current Behavior:**
   - I(t) = I₀ × sin(2π × f × t)
   - I₀ = current amplitude (peak current)
   - f = frequency in Hz
   - t = time in seconds"""

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
def test_hypothesis(current1, current2, distance):
	C = 4.78e-2
	return (C * current1 * current2) / (distance ** 3)

# Test with different parameters
test_I1 = [1.0, 2.0, 3.0]
test_I2 = [1.0, 1.0, 1.0]
test_distance = [1.0, 2.0, 3.0]

for I1_val, I2_val, d_val in zip(test_I1, test_I2, test_distance):
	force = test_hypothesis(I1_val, I2_val, d_val)
	print(f"I1={I1_val}, I2={I2_val}, d={d_val} → F={force}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
I1=1.0, I2=1.0, d=1.0 → F=0.0478
I1=2.0, I2=1.0, d=2.0 → F=0.0060
I1=3.0, I2=1.0, d=3.0 → F=0.0018

**Your Code:**
```python
def test_hypothesis(current1, current2, distance):
	C = 4.78e-2
	return (C * current1 * current2) / (distance ** 3)

# Test with different parameters
test_I1 = [1.0, 2.0, 3.0]
test_I2 = [1.0, 1.0, 1.0]
test_distance = [1.0, 2.0, 3.0]

for I1_val, I2_val, d_val in zip(test_I1, test_I2, test_distance):
	force = test_hypothesis(I1_val, I2_val, d_val)
	print(f"I1={I1_val}, I2={I2_val}, d={d_val} → F={force}")
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
		prompts.append(LINEAR_MOTION_PROMPT.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))
	elif system == ExperimentSystem.COMPLEX_SYSTEM:
		prompts.append(COMPLEX_SYSTEM_PROMPT.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))

	# Add code assisted instructions if requested
	if is_code_assisted:
		prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
	
	prompts.append(SUBMISSION_REQUIREMENTS.format(
		function_signature=FUNCTION_SIGNATURE,
		return_description=RETURN_DESCRIPTION,
		example=EXAMPLE
	))
	return "\n\n".join(prompts)