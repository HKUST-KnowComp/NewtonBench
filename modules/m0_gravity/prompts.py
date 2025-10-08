from .m0_types import ExperimentSystem
from modules.common.prompts_base import (
	OBJECTIVE_PROMPT,
	ASSISTING_LAWS_DISCLAIMER,
	SUBMISSION_REQUIREMENTS,
	RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
	RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- mass1: mass of the first object. It should be a positive real number.
- mass2: mass of the second object. It should be a positive real number.
- distance: distance between the two objects. It should be a positive real number."""

# Gravity-specific submission requirements
FUNCTION_SIGNATURE = "def discovered_law(mass1, mass2, distance):"
RETURN_DESCRIPTION = "the magnitude of the gravitational force acting on mass2 by mass1"
EXAMPLE = """**Example 1:**
<final_law>
def discovered_law(mass1, mass2, distance):
   import math
   C = 8e-8
   return (C * mass1 * mass2) / (math.pow(distance, 2))
</final_law>

**Example 2:**
<final_law>
def discovered_law(mass1, mass2, distance):
   C = 8e-8
   return (C * mass1 * mass2) / (distance ** 2)
</final_law>"""

# Vanilla Equation Prompt
VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a device that can position two objects and measure the force acting on the second object (mass2) by the first object (mass1). You have precise control over the following properties for each experiment you run:
- Mass of the first object (`mass1`)
- Mass of the second object (`mass2`)
- Distance between the objects' centers (`distance`)

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"mass1": ..., "mass2": ..., "distance": ...}},
  {{"mass1": ..., "mass2": ..., "distance": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of the measured force.
<experiment_output>
[6.673e-05, 6.675e-04]
</experiment_output>"""

# Orbital motion discovery prompt
ORBITAL_MOTION_PROMPT = """**Experimental Apparatus:**
You have access to a 2D motion tracking system that can:
1. Fix one mass (mass1) at the origin (0,0)
2. Place a second mass (mass2) at any initial position
3. Give the second mass an initial velocity
4. Track the position and velocity of mass2 over time

**Control Parameters:**
- `mass1`: Mass of the fixed central object
- `mass2`: Mass of the orbiting object
- `distance`: Starting distance from origin
- `initial_velocity`: Initial velocity magnitude (perpendicular to radius)
- `duration`: Time to track motion
- `time_step`: Time interval between measurements

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"mass1": ..., "mass2": ..., "distance": ..., "initial_velocity": ..., "duration": ..., "time_step": ...}},
   {{"mass1": ..., "mass2": ..., "distance": ..., "initial_velocity": ..., "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "position": [...], "velocity": [...]}} ,
   {{"time": [...], "position": [...], "velocity": [...]}}
]
</experiment_output>

**Physics Background:**
- The motion is confined to a 2D plane
- The force acts along the line connecting the masses

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Newton's Second Law: F = ma
   - Force equals mass times acceleration
   - Acceleration vector points in the same direction as the force vector

2. Energy Conservation: E = ½mv² + V(r)
   - Total mechanical energy remains constant
   - Kinetic energy: KE = ½mv²
   - Potential energy: V(r) is a function of distance we want to discover
   
3. For circular orbits (if they exist):
   - Centripetal force: F = mv²/r
   - Orbital period: T = 2πr/v

4. Kinematics Equations:
   - v = v₀ + at
   - x = x₀ + v₀t + ½at²
   - v² = v₀² + 2a(x - x₀)
   These equations relate position, velocity, and acceleration"""

# Linear motion discovery prompt
LINEAR_MOTION_PROMPT = """**Experimental Apparatus:**
You have access to a 1D motion tracking system that can:
1. Fix one mass (mass1) at the origin (x=0)
2. Place a second mass (mass2) at any initial position on the x-axis
3. Give mass2 an initial velocity along the x-axis
4. Track the position and velocity of mass2 over time

**Control Parameters:**
- `mass1`: Mass of the fixed object at origin
- `mass2`: Mass of the moving object
- `distance`: Starting position on x-axis
- `initial_velocity`: Initial velocity of the moving mass
- `duration`: Time to track motion
- `time_step`: Time interval between measurements

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments in side the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
   {{"mass1": ..., "mass2": ..., "distance": ..., "initial_velocity": ..., "duration": ..., "time_step": ...}},
   {{"mass1": ..., "mass2": ..., "distance": ..., "initial_velocity": ..., "duration": ..., "time_step": ...}}
]
</run_experiment>

**System Response:**
The system will return a list of time series data objects (at most 20 sets of data per experiment):
<experiment_output>
[
   {{"time": [...], "position": [...], "velocity": [...]}} ,
   {{"time": [...], "position": [...], "velocity": [...]}}
]
</experiment_output>

**Physics Background:**
- Motion is restricted to one dimension
- The force acts along the x-axis

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1. Newton's Second Law: F = ma
   - Force equals mass times acceleration
   - Since motion is 1D, all vectors are scalars along the x-axis

2. Kinematics Equations:
   - v = v₀ + at
   - x = x₀ + v₀t + ½at²
   - v² = v₀² + 2a(x - x₀)
   These equations relate position, velocity, and acceleration

3. Energy Conservation:
   - Total mechanical energy remains constant
   - Work-Energy Theorem: W = F·d = ΔKE = ½m(v² - v₀²)
   - This relates force to changes in kinetic energy"""

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
def test_hypothesis(mass1, mass2, distance):
	# Test your hypothesis: F = C * m1 * m2 / r^2
	C = 6.674e-5
	return C * mass1 * mass2 / (distance ** 2)

# Test with different parameters
test_mass1 = [1.0, 2.0, 3.0]
test_mass2 = [1.0, 1.0, 1.0]
test_distance = [1.0, 2.0, 3.0]

for m1, m2, d in zip(test_mass1, test_mass2, test_distance):
	force = test_hypothesis(m1, m2, d)
	print(f"m1={m1}, m2={m2}, d={d} → F={force}")
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
m1=1.0, m2=1.0, d=1.0 → F=6.674e-05
m1=2.0, m2=1.0, d=2.0 → F=3.337e-05
m1=3.0, m2=1.0, d=3.0 → F=2.225e-05

**Your Code:**
```python
def test_hypothesis(mass1, mass2, distance):
	# Test your hypothesis: F = C * m1 * m2 / r^2
	C = 6.674e-5
	return C * mass1 * mass2 / (distance ** 2)

# Test with different parameters
test_mass1 = [1.0, 2.0, 3.0]
test_mass2 = [1.0, 1.0, 1.0]
test_distance = [1.0, 2.0, 3.0]

for m1, m2, d in zip(test_mass1, test_mass2, test_distance):
	force = test_hypothesis(m1, m2, d)
	print(f"m1={m1}, m2={m2}, d={d} → F={force}")
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
		prompts.append(LINEAR_MOTION_PROMPT.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))
	elif system == ExperimentSystem.COMPLEX_SYSTEM:
		prompts.append(ORBITAL_MOTION_PROMPT.format(ASSISTING_LAWS_DISCLAIMER=ASSISTING_LAWS_DISCLAIMER, RUN_EXPERIMENT_INSTRUCTION=run_experiment_instruction))

	# Add code assisted instructions if requested
	if is_code_assisted:
		prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
	
	prompts.append(SUBMISSION_REQUIREMENTS.format(
		function_signature=FUNCTION_SIGNATURE,
		return_description=RETURN_DESCRIPTION,
		example=EXAMPLE
	))
	return "\n\n".join(prompts) 