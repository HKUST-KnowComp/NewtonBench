import numpy as np
import math
from typing import Callable, Any, Dict
import re
import ast
import inspect
from utils.call_llm_api import call_llm_api
from .prompts_base import SYMBOLIC_EQUIVALENCE_JUDGE_PROMPT


def extract_formula_from_function(func: Callable):
    """
    Extract the formula from a ground truth law function.
    Args:
        func: The ground truth law function object.
    Returns:
        formula_str: The formula as a string
    """
    # Get source code
    src = inspect.getsource(func)
    # Parse AST
    tree = ast.parse(src)
    func_def = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if func_def is None:
        raise ValueError("No function definition found in source.")
    # Find all return statements
    return_nodes = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    if not return_nodes:
        raise ValueError("No return statement found in function.")
    # Resolve each and collect non-constant candidates
    candidates = []
    for rn in return_nodes:
        resolved = _resolve_wrapped_expression(rn.value, func_def, getattr(rn, 'lineno', 0))
        if not isinstance(resolved, ast.Constant):
            candidates.append(resolved)
    if not candidates:
        raise ValueError("No non-constant return found.")
    # Pick the first non-constant (assuming it's the main formula)
    core_expr = candidates[0]
    formula_str = ast.unparse(core_expr)
    return formula_str

def _resolve_wrapped_expression(expr: ast.AST, func_def: ast.FunctionDef, return_lineno: int) -> ast.AST:
    """
    Attempt to unwrap simple wrappers around the core mathematical expression so
    the judge sees the actual formula instead of helper constructs.
    - float(value) -> value
    - return of a variable -> resolve latest assignment to that variable prior to return
    """
    # Unwrap float(<expr>) calls
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'float' and len(expr.args) == 1:
        return _resolve_wrapped_expression(expr.args[0], func_def, return_lineno)
    # If returning a variable name, try to find its latest assignment before the return
    if isinstance(expr, ast.Name):
        var_name = expr.id
        last_assigned_expr = None
        last_assigned_lineno = -1
        for node in ast.walk(func_def):
            node_lineno = getattr(node, 'lineno', 0)
            if node_lineno and node_lineno < return_lineno:
                if isinstance(node, ast.Assign):
                    if (
                        len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == var_name
                        and node_lineno > last_assigned_lineno
                    ):
                        last_assigned_expr = node.value
                        last_assigned_lineno = node_lineno
                elif isinstance(node, ast.AnnAssign):
                    if (
                        isinstance(node.target, ast.Name)
                        and node.target.id == var_name
                        and node.value is not None
                        and node_lineno > last_assigned_lineno
                    ):
                        last_assigned_expr = node.value
                        last_assigned_lineno = node_lineno
        if last_assigned_expr is not None:
            return _resolve_wrapped_expression(last_assigned_expr, func_def, return_lineno)
        # Fallback to original name if no assignment found
        return expr
    # If expression is a conditional (a if cond else b), prefer the non-NaN branch heuristically
    if isinstance(expr, ast.IfExp):
        def is_nan_literal(e: ast.AST) -> bool:
            # Matches float('nan')
            return (
                isinstance(e, ast.Call)
                and isinstance(e.func, ast.Name)
                and e.func.id == 'float'
                and len(e.args) == 1
                and isinstance(e.args[0], ast.Constant)
                and isinstance(e.args[0].value, str)
                and e.args[0].value.lower() == 'nan'
            )
        # Prefer body if it isn't NaN, else orelse
        if not is_nan_literal(expr.body):
            return _resolve_wrapped_expression(expr.body, func_def, return_lineno)
        return _resolve_wrapped_expression(expr.orelse, func_def, return_lineno)
    return expr

def calculate_rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE) between true and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)  
    # Mask to filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_true_positive = np.abs(y_true)
    y_pred_positive = np.abs(y_pred)
    return float(np.sqrt(np.nanmean((np.log1p(y_pred_positive) - np.log1p(y_true_positive))**2)))

def calculate_exact_accuracy(symbolic_equivalent: bool) -> float:
    return 1.0 if symbolic_equivalent else 0.0

def add_necessary_imports(function_str: str) -> str:
    """
    Automatically add necessary imports to the LLM's function string.
    """
    imports_needed = []
    
    # Check for math functions
    math_functions = ['math.pow', 'math.exp', 'math.sqrt', 'math.log', 'math.sin', 'math.cos', 'math.tan']
    if any(func in function_str for func in math_functions):
        imports_needed.append('import math')
    
    # Check for numpy functions
    numpy_functions = ['np.', 'numpy.']
    if any(func in function_str for func in numpy_functions):
        imports_needed.append('import numpy as np')
    
    # Add imports at the beginning if needed
    if imports_needed:
        import_lines = '\n'.join(imports_needed)
        # Find the function definition line
        lines = function_str.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def discovered_law'):
                # Insert imports before the function definition
                lines.insert(i, import_lines)
                break
        return '\n'.join(lines)
    
    return function_str

def llm_symbolic_equivalence_judge(llm_formula_str: str, gt_formula_str: str, param_description: str, judge_model_name: str = "nemotron-ultra", trial_info=None) -> bool:
    """
    Use LLM to determine if two formulas are mathematically equivalent.
    Allows up to 3 retries if there is any error or if the answer cannot be matched.
    
    Args:
        llm_formula_str: String representation of LLM's formula
        gt_formula_str: String representation of ground truth formula
        param_description: String representation of the parameter descriptions
        judge_model_name: name of model as LLM judge
        trial_info: Optional trial information dictionary
        
    Returns:
        Boolean indicating if formulas are mathematically equivalent
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Format the prompt with the two equations
            prompt = SYMBOLIC_EQUIVALENCE_JUDGE_PROMPT.format(
                equation1=gt_formula_str,
                equation2=llm_formula_str,
                param_description=param_description
            ) 

            messages = [{"role": "system", 'content': "detailed thinking on"}] if "nemotron" in judge_model_name else []
            messages.append({"role": "user", "content": prompt})
            response, reasoning_response, _ = call_llm_api(messages, model_name=judge_model_name, temperature=0.6, trial_info=trial_info)
            
            if response is None:
                print(f"[LLM Judge] Attempt {attempt}: No response received. Retrying...")
                continue
                
            # Parse the response to extract YES/NO from "Answer: YES/NO" format
            response_upper = response.strip().upper()
            
            # Look for "Answer:" followed by YES or NO
            answer_match = re.search(r'ANSWER:\s*(YES|NO)', response_upper)
            if answer_match:
                answer = answer_match.group(1)
                return answer == "YES"
                         
            # Fallback: search for the last occurrence of YES or NO
            all_matches = re.findall(r'\b(YES|NO)\b', response_upper)
            if all_matches:
                last_answer = all_matches[-1]
                return last_answer == "YES"
            else:
                print(f"[LLM Judge] Attempt {attempt}: Could not find 'Answer: YES/NO' in response. Retrying...")
                continue
        except Exception as e:
            print(f"[LLM Judge] Attempt {attempt}: Exception occurred: {e}. Retrying...")
            continue
    print(f"[LLM Judge] All {max_retries} attempts failed. Returning False.")
    return False

def evaluate_law(
    llm_function_str: str,
    gt_law: Callable,
    test_data: Dict[str, np.ndarray],
    parameter_mapping: Dict[str, str],
    param_description: str,
    judge_model_name: str = "nemotron-ultra",
    trial_info: Any = None,
    symbolic_check: bool = True
) -> dict:
    """
    Generic evaluation function for LLM-discovered law.
    
    Args:
        llm_function_str: String containing the LLM's function definition
        gt_law: Ground truth function to compare against
        test_data: Dictionary containing test data arrays
        parameter_mapping: Dict mapping function parameter names to test_data keys
                          e.g., {"mass1": "mass1", "mass2": "mass2", "distance": "distance"}
        param_description: String representation of the parameter descriptions
        judge_model_name: Name of the LLM model to use for symbolic equivalence checking
        trial_info: Trial information for logging
        symbolic_check: Whether to perform symbolic equivalence checking
    """
    symbolic_equivalent = False
    symbolic_msg = None
    rmsle = float("nan")
    error_msg = None
    formula_string = ""
    try:
        # Add necessary imports to the LLM's function
        fixed_function_str = add_necessary_imports(llm_function_str)
        
        # Execute the LLM's code to define the function
        local_scope = {}
        exec(fixed_function_str, globals(), local_scope)
        llm_function = local_scope.get('discovered_law')
        
        # Get the number of test points
        first_key = list(parameter_mapping.values())[0]
        num_points = len(test_data[first_key])
        
        # Generate ground truth and predicted values using parameter mapping
        y_true = np.array([
            gt_law(*[test_data[param_key][i] for param_key in parameter_mapping.values()])
            for i in range(num_points)
        ])
        y_pred = np.array([
            llm_function(*[test_data[param_key][i] for param_key in parameter_mapping.values()])
            for i in range(num_points)
        ])
        
        rmsle = calculate_rmsle(y_true, y_pred)
    except Exception as e:
        error_msg = f"Failed to execute or evaluate submitted code: {e}"
        
    try:
        if symbolic_check:
            # Create default trial_info if none provided
            if trial_info is None:
                trial_info = {'trial_id': 'llm_judge'}
            formula_string = extract_formula_from_function(gt_law)
            symbolic_equivalent = llm_symbolic_equivalence_judge(
                llm_formula_str=llm_function_str,
                gt_formula_str=formula_string,
                param_description=param_description,
                judge_model_name = judge_model_name,
                trial_info=trial_info
            )
            symbolic_msg = None if symbolic_equivalent else "LLM judge determined formulas are not equivalent."
    except Exception as e:
        symbolic_msg = f"Symbolic equivalence check failed: {e}"
    exact_accuracy = calculate_exact_accuracy(symbolic_equivalent)
    return {
        "rmsle": float(rmsle),
        "ground_truth_law": formula_string,
        "exact_accuracy": float(exact_accuracy),
        "symbolic_equivalent": symbolic_equivalent,
        "symbolic_msg": symbolic_msg,
        "error": error_msg
    } 