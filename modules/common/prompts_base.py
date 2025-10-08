# Base prompt for all modes (shared by all modules)
OBJECTIVE_PROMPT = """**Mission:** Your objective is to act as an AI research physicist. You are in a simulated universe and your goal is to discover the physical law in this universe.
Note that the laws of physics in this universe may differ from those in our own, including both factor dependency and constant scalars.
There's no ground-truth laws available to you. You must discover the law yourself."""

# Disclaimer for assisting laws (shared by all modules)
ASSISTING_LAWS_DISCLAIMER = """**Important Note About Physics Laws:**
Only the assisting laws listed below are guaranteed to hold true in this simulated universe. Other physics laws from our universe may or may not apply. You should rely primarily on experimental data and these confirmed laws to discover the underlying force law."""

RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE = """**How to Run Experiments:**
To gather data, you must use the <run_experiment> tag. Provide a JSON array specifying the parameters for one or arbitrarily many experimental sets. Note that all measurements returned by the system are **noise-free**. You can assume the data is perfectly accurate and deterministic."""

RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE = """**How to Run Experiments:**
To gather data, you must use the <run_experiment> tag. Provide a JSON array specifying the parameters for one or arbitrarily many experimental sets. All measurements returned by the system are subject to **random noise**, simulating the imperfections of real-world sensors."""

# Need to add format {} so that this can be dynamically changed for all modules, including function signature, examples, etc.
# Common submission requirements for all modes (shared by all modules)
SUBMISSION_REQUIREMENTS = """**Final Submission:**
Once you are confident you have determined the underlying force law, submit your findings as a single Python function enclosed in <final_law> tags.

**Submission Requirements:**
1. The function must be named `discovered_law`
2. The function signature must be exactly: `{function_signature}`
3. The function should return {return_description}.
4. If you conclude that one of these parameters does not influence the final force, you should simply ignore that variable within your function's logic rather than changing the signature.
5. If your law contains any constants, you must define the constant as a local variable inside the function body. Do NOT include the constant as a function argument.
6. Import any necessary libraries inside the function body (e.g. math, numpy, etc.) if needed

**Critical Boundaries:**
- Do NOT include any explanation or commentary inside the <final_law> blocks and the function body.
- Only output the <final_law> block in your final answer.

{example}

**Reminder:**
1. Always remember that the laws of physics in this universe may differ from those in our own, including factor dependency, constant scalars, and the form of the law.
2. When doing the experiments, use a broad range of input parameters, for example, values spanning from 10^-3 to 10^15 to ensure robustness across scales."""

# LLM Judge Prompt for Symbolic Equivalence (shared by all modules)
SYMBOLIC_EQUIVALENCE_JUDGE_PROMPT = """You are a mathematical judge. Your task is to determine if two equations are equivalent.

**Instructions:**
1. Compare the two equations carefully
2. Consider algebraic manipulations, variable reordering, and variable renaming
3. Determine if they represent the same mathematical relationship
4. Provide your reasoning step by step first, and then provide only one answer under the format of **Answer: YES/NO**
5. Try converting both equations into the same algebraic form to make comparison easier.
   - e.g. rewrite ln(x ** 2) into 2ln(x)

**Output format:**
Reasoning: (Your reasoning steps)
Answer: (YES/NO)

**Reminder:**
- Equations may be expressed in standard mathematical notation or as Python code. If the Python implementation implies the same mathematical relationship, the equations are considered equivalent.
- Constants may differ in form or value. As long as they serve the same functional role (e.g., both scale the output proportionally), they are considered interchangeable.
   - For example, a constant expressed as sqrt(k) in one equation and as c in another may be equivalent if both affect the output in the same way and can be interchangeable by selecting suitable value for the constant
- Variable names may differ, but the index and structure of variables must match exactly for the equations to be considered equivalent.
   - For example, index of 4 and 4.03 are considered different
- YES/NO must be on the same line as "Answer:"

**Examples:**

Equation 1: (HIDDEN_CONSTANT_C * x1 * x2) ** 2 / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   C = 6.7e-05
   return (C * (x1 * x2) ** 2) / x3 ** 2
Reasoning: Although the constant in equation 1 is HIDDEN_CONSTANT_C**2 and constant in equation 2 is C, both constant serve the same scaling role ......
Answer: YES

Equation 1: (C * x1 * x2) / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   C = 6.7e-05
   return (C * x1) / (x3 ** 4 * x2)
Reasoning: The second equation changes the exponent on x3 and alters the position of x2 ......
Answer: NO

Equation 1: sqrt(C * x1 * (x2 ** 2)) / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   C = 6.7e-05
   return sqrt(C * x1) * x2 / x3 ** 2
Reasoning: Since sqrt(x2 ** 2) = x2, both expressions represent same mathematical relationship ......
Answer: YES

Equation 1: (G * x1 * x2) / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   C = 6.7e-05
   return (C * x1 * x2) / x3 ** 2.02
Reasoning: The exponent on x3 differs slightly ......
Answer: NO

Equation 1: (C * x1 * x2) / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   G = 6.7e-05
   product = x1 * x2
   return (G * product) / x3 ** 2
Reasoning: Variable naming differs but structure and operations are equivalent. G serves the same role as C ......
Answer: YES

Equation 1: C * ln(x ** 2)
Equation 2: def discovered_law(x1, x2, x3):
   G = 2.02
   return G * ln(x)
Reasoning: C * ln(x**2) is the same as 2C * ln(x) and the constant (2C) servers the same role as G ......
Answer: YES

Equation 1: (C * x1 * x2) / x3 ** 2
Equation 2: def discovered_law(x1, x2, x3):
   return (x1 * x2) / x3 ** 2
Reasoning: Equation 1 has a constant variable while Equation 2 has a numerical constant of 1 ......
Answer: YES

**Your Task:**
Compare these two equations and determine if they are equivalent:

Parameter Descriptions:
{param_description}

Equation 1: {equation1}

Equation 2: {equation2}""" 