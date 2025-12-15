from .m10_types import ExperimentSystem
from modules.common.prompts_base import (
	OBJECTIVE_PROMPT,
	ASSISTING_LAWS_DISCLAIMER,
	SUBMISSION_REQUIREMENTS,
	RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
	RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE
)

PARAM_DESCRIPTION = """- omega: angular frequency. It should be a positive real number.
- T: temperature. It should be a positive real number."""

# --- Core Law Discovery ---
FUNCTION_SIGNATURE = "def discovered_law(omega, T):"
RETURN_DESCRIPTION = "the average occupation number of photons (n) in quantum state"
EXAMPLE = """**Example:**
<final_law>
def discovered_law(omega, T):
	import math
	Constant = 1.4314312-21 
	return 1 / (math.exp(Constant * omega / T) - 1)
</final_law>

**Note**:
- `omega` is the angular frequency of the photons.
- `T` is the temperature."""

VANILLA_EQUATION_PROMPT = """**Experimental Apparatus:**
You have access to a photons analyzer that can directly measure the average occupation number `n` of photons in quantum state for a system.

**Control Parameters:**
- `omega`: The angular frequency of the photons.
    - the scale should be at least 1e8
    - remember to test among different scales of `omega`
- `temperature`: The temperature of the system.
    - the scale should be at least 1e1
    - remember to test among different scales of `temperature`

{RUN_EXPERIMENT_INSTRUCTION}

**CRITICAL: Strict Format Adherence**
- You MUST use the exact <run_experiment> tag format shown below.
- If your format is incorrect, the system will ask you to read the initial prompt again.
- Double-check your JSON syntax before submitting.
- Ensure all required parameters are included with correct names.

**Input/Output Format:**
You must use the following JSON format for your requests and don't add any comments inside the JSON. The system will respond with a corresponding output array.

*Your Request:*
<run_experiment>
[
  {{"omega": ..., "temperature": ...}},
  {{"omega": ..., "temperature": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measured average occupation numbers (n).
<experiment_output>
[..., ...]
</experiment_output>

**Strategy**: Analyze the functional roles of parameters in relation to the average occupation number of photons:
- **Similar roles**: Identify parameters that influence the average occupation number of photons in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system.
- The relationship you are trying to discover is highly non-linear (e.g. containing exponential function, trigonometric function, etc) and sensitive to the scale of your inputs. A robust experimental strategy is essential.
    - **Explore Orders of Magnitude:** Probing parameters at very different scales is critical. Small, incremental changes to your inputs will likely yield very similar, uninformative outputs. **Your goal is to map the function's behavior across its entire dynamic range.**
    - **The Law is Not a Simple Constant:** Be aware that the occupation number `n` is a strong function of both `omega` and `T`. If your experiments consistently return a constant value (e.g., always near zero or a very large number), it is a sign that your tests are confined to an asymptotic regime (a flat part of the curve). This means you must drastically change your parameters to find the interesting, transitional behavior.
        - Please remember that the law will **NEVER** be a simple constant
    - **Combine Parameters:** The physics of this system depends on how `omega` and `T` relate to each other. After exploring each parameter individually, design experiments where you change both simultaneously. For example, investigate if doubling `omega` can be compensated for by a change in `T`."""

# --- Simple System: The Black-Body Spectrometer ---
SIMPLE_SYSTEM_PROMPT = """**Experimental Apparatus:**
You have access to a black-body cavity and a high-precision radiometer. You can set the temperature of the cavity and then use the radiometer to measure the emitted power at a specific frequency.

**Experimental Setup:**
1.  **Black-Body Cavity**: An idealized object that perfectly absorbs all incident electromagnetic radiation and emits thermal radiation based on its temperature.
2.  **Tunable Radiometer**: An instrument that measures the power of radiation (spectral radiance) for a specific, user-defined frequency.

**Control Parameters:**
- `temperature`: The absolute temperature (`T`) of the cavity.
    - the scale should be at least 1e1
    - remember to test among different scales of `T`
- `probe_frequency`: The specific angular frequency (`ω`) at which to measure the radiation.
    - the scale should be at least 1e8
    - remember to test among different scales of `probe_frequency`

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
  {{"temperature": ..., "probe_frequency": ...}},
  {{"temperature": ..., "probe_frequency": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the spectral radiance.
<experiment_output>
[
  {{"spectral_radiance": "..."}},
  {{"spectral_radiance": "..."}}
]
</experiment_output>

**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following laws are guaranteed to hold in this universe:
1.  **Spectral Radiance and Occupation Number:**
    - The measured `spectral_radiance` (`R`) at a frequency (`ω`) is proportional to the number of photons (`n`) at that frequency and their energy. 
    - The exact relationship is `R(ω) ∝ n(ω) * ω³`.

2.  **Stefan-Boltzmann Law:** 
    - The total power radiated per unit area is proportional to the fourth power of the temperature (`P/A ∝ T^4`).

**Strategy**: Analyze the functional roles of parameters in relation to the average occupation number of photons:
- **Similar roles**: Identify parameters that influence the average occupation number of photons in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system.
- The relationship you are trying to discover is highly non-linear (e.g. containing exponential function, trigonometric function, etc) and sensitive to the scale of your inputs. A robust experimental strategy is essential.
    - **Explore Orders of Magnitude:** Probing parameters at very different scales is critical. Small, incremental changes to your inputs will likely yield very similar, uninformative outputs. **Your goal is to map the function's behavior across its entire dynamic range.**
    - **The Law is Not a Simple Constant:** Be aware that the occupation number `n` is a strong function of both `omega` and `T`. If your experiments consistently return a constant value (e.g., always near zero or a very large number), it is a sign that your tests are confined to an asymptotic regime (a flat part of the curve). This means you must drastically change your parameters to find the interesting, transitional behavior.
        - Please remember that the law will **NEVER** be a simple constant
    - **Combine Parameters:** The physics of this system depends on how `omega` and `T` relate to each other. After exploring each parameter individually, design experiments where you change both simultaneously. For example, investigate if doubling `omega` can be compensated for by a change in `T`."""

# --- Complex System: The Photon Gas Compressor ---
COMPLEX_SYSTEM_PROMPT = """**Experimental Apparatus:**
You are using a calorimeter to measure the radiation from a black-body cavity. A tunable frequency filter is placed between the cavity and the calorimeter, allowing only a specific band of frequencies to pass through and be measured.

**Experimental Setup:**
1.  **Black-Body Cavity**: An idealized object that perfectly absorbs all incident electromagnetic radiation and emits thermal radiation based on its temperature.
2.  **Tunable Band-Pass Filter**: A filter defined by a `center_frequency` and a `bandwidth`.
3.  **Calorimeter**: A device that absorbs all incoming radiation and measures the total power received.

**Control Parameters:**
- `temperature`: The absolute temperature (`T`) of the cavity.
    - the scale should be at least 1e1
    - remember to test among different scales of `T`
- `center_frequency`: The central angular frequency (`ω_c`) that the filter is tuned to.
    - the scale should be at least 1e8
    - remember to test among different scales of `center_frequency`
- `bandwidth`: The width of the frequency band (`Δω`) that the filter allows to pass.

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
  {{"temperature": ..., "center_frequency": ..., "bandwidth": ...}},
  {{"temperature": ..., "center_frequency": ..., "bandwidth": ...}}
]
</run_experiment>

*System Response:*
The system will return a list of measurement objects, each containing the total power that passed through the filter.
<experiment_output>
[
  {{"total_power": "..."}},
  {{"total_power": "..."}}
]
</experiment_output>
	
**Confirmed Assisting Laws:**
{ASSISTING_LAWS_DISCLAIMER}

The following are guaranteed to hold in this universe:
1.  **Power and Spectral Radiance:**
    - The measured `total_power` (`P`) is the integral of the spectral radiance (`R(ω)`) over the frequency range allowed by the filter.
    - `P = ∫ R(ω) dω` (from `ω_c - Δω/2` to `ω_c + Δω/2`).

2.  **Spectral Radiance and Occupation Number:**
    - The measured `spectral_radiance` (`R`) at a frequency (`ω`) is proportional to the number of photons (`n`) at that frequency and their energy. 
    - The exact relationship is `R(ω) ∝ n(ω) * ω³`.
    
**Strategy**: Analyze the functional roles of parameters in relation to the average occupation number of photons:
- **Similar roles**: Identify parameters that influence the average occupation number of photons in comparable ways (e.g., both increasing or decreasing it proportionally).
- **Different roles** Identify parameters that affect fundamentally different aspects of the system.
- The relationship you are trying to discover is highly non-linear (e.g. containing exponential function, trigonometric function, etc) and sensitive to the scale of your inputs. A robust experimental strategy is essential.
    - **Explore Orders of Magnitude:** Probing parameters at very different scales is critical. Small, incremental changes to your inputs will likely yield very similar, uninformative outputs. **Your goal is to map the function's behavior across its entire dynamic range.**
    - **The Law is Not a Simple Constant:** Be aware that the occupation number `n` is a strong function of both `omega` and `T`. If your experiments consistently return a constant value (e.g., always near zero or a very large number), it is a sign that your tests are confined to an asymptotic regime (a flat part of the curve). This means you must drastically change your parameters to find the interesting, transitional behavior.
        - Please remember that the law will **NEVER** be a simple constant
    - **Combine Parameters:** The physics of this system depends on how `omega` and `T` relate to each other. After exploring each parameter individually, design experiments where you change both simultaneously. For example, investigate if doubling `omega` can be compensated for by a change in `T`.
- **Approximating the Integral:** Your instrument measures `total_power` (`P`), which is the integral of the `spectral_radiance` (`R(ω)`). You cannot measure `R(ω)` directly, but you have control over the integration interval via the `bandwidth` (`Δω`) parameter. Consider the relationship between an integral and the function itself when the integration interval (`Δω`) is made very narrow so that you can find a approximation function of the power other than directly calculating the integration"""

# Code assisted specific instructions for interactive Python code execution
CODE_ASSISTED_PROMPT_INSTRUCTION = """**IMPORTANT: You have access to interactive Python code execution through <python> tags.**

**How to use <python> tags:**
1. **You can write ANY Python code** - functions, calculations, print statements, etc.
2. **Format:** <python>your_python_code_here</python>
3. **You can submit multiple <python> tags** to test different ideas.
4. **Each <python> tag will be executed** and you'll get immediate feedback in <python_output> tags.
5. **Use the feedback** to refine your understanding and calculations.

**CRITICAL: Use EXACTLY these tags:**
- Start with: `<python>`
- End with: `</python>`
- NOT `<|python_tag|>` or any other variation.

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
def test_hypothesis(omega, T):
	import math
	Constant = 1.4314312-21 
	try:
		return 1 / (math.exp(Constant * omega / T) - 1)
	except (ValueError, ZeroDivisionError):
		return float('nan')

# Test with some plausible values
print(test_hypothesis(omega=2e14, T=5000))
</python>
```

**System Response:**
```
<python_output>
✅ **Python Code Execution Successful!**

**Output:**
1.51421e-3

**Your Code:**
```python
def test_hypothesis(omega, T):
	import math
	Constant = 1.4314312-21 
	try:
		return 1 / (math.exp(Constant * omega / T) - 1)
	except (ValueError, ZeroDivisionError):
		return float('nan')

# Test with some plausible values
print(test_hypothesis(omega=2e14, T=5000))
```
</python_output>
```

**Data-driven discovery of exponents:**
```
<python>
import numpy as np

# Suppose you have collected this data from the Photon Gas Compressor
# T (K): [3000, 4000, 5000, 6000]
# P (Pa): [6.1, 19.3, 47.2, 97.8]

T_data = np.array([3000, 4000, 5000, 6000])
P_data = np.array([6.1, 19.3, 47.2, 97.8])

# We suspect a power-law relationship: P = C * T^a
# To find the exponent 'a', we can use linear regression on the log-transformed equation:
# log(P) = log(C) + a*log(T)

log_T = np.log(T_data)
log_P = np.log(P_data)

# Fit a line (degree 1 polynomial) to the log-log data
# The slope of this line is the exponent 'a'
coeffs = np.polyfit(log_T, log_P, 1)
a = coeffs[0]

print(f"Found exponent (a): {a:.2f}")
print("This suggests the law is P ∝ T^4.0")
</python>
```

**Workflow:**
1. **Analyze the problem** and form initial hypotheses based on the provided information and assisting laws.
2. **Use <python> tags** to test your ideas, perform calculations, or analyze data from experiments.
3. **Analyze the results** from <python_output> and refine your understanding.
4. **Repeat** with more <python> tags until you're confident in your solution.
5. **Submit final law** using <final_law> tags with the proper Python function format."""

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
	
	if is_code_assisted:
		prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)
	
	prompts.append(SUBMISSION_REQUIREMENTS.format(
		function_signature=FUNCTION_SIGNATURE,
		return_description=RETURN_DESCRIPTION,
		example=EXAMPLE
	))
	
	return "\n\n".join(prompts)
