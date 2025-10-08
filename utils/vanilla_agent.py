# utils/vanilla_agent.py

import re
from typing import List, Dict, Any, Tuple
import json

from utils.call_llm_api import call_llm_api

# --- Base Prompt ---
BASE_PROMPT = """You are an AI research assistant tasked with discovering scientific laws in a simulated universe.
Your goal is to propose experiments, analyze the data they return, and ultimately deduce the underlying scientific law.
Please note that the laws of physics in this universe may differ from those in our own.
You can perform experiments to gather data but you must follow the protocol strictly.

**Workflow:**
1.  Analyze the mission description provided.
2.  Design a set of experiments to test your hypotheses.
3.  Use the `<run_experiment>` tag to submit your experimental inputs.
4. The system will return the results (up to 20 data points per experiment) in an <experiment_output> tag.
    - If a returned value is nan, it indicates that the calculation encountered an error, such as:
        - ValueError (e.g., using asin on a value outside the valid range of [-1, 1])
        - OverflowError (e.g., using exp on an extremely large input)
    - You may ignore any data points that return nan, as they do not contribute to valid hypothesis testing.
    - Consider adjusting your input parameters to avoid invalid ranges and improve data coverage.
5.  You can run up to {max_turns} rounds of experiments. Use them wisely so that before submitting your final law, ensure you have:
    - fully explored the experimental space
    - Verified your hypotheses against the data
    - made the most of the available rounds to strengthen your conclusions
6.  Only one action is allowed per round: either <run_experiment> or <final_law>.
7.  After submitting <run_experiment>, wait for <experiment_output> before proceeding.
8.  You should verify your hypotheses by checking if the output from the experiments matches the output from your hypotheses.
9.  When confident, submit your final discovered law using the `<final_law>` tag. This ends the mission."""

def parse_experiment_request(response_text: str) -> List[Dict[str, float]]:
    """Parses the LLM's requested experiments from the <run_experiment> block (expects JSON array)."""
    start_tag = '<run_experiment>'
    end_tag = '</run_experiment>'
    
    start_index = response_text.rfind(start_tag)
    if start_index == -1:
        return []
        
    end_index = response_text.find(end_tag, start_index)
    if end_index == -1:
        return []

    content = response_text[start_index + len(start_tag):end_index].strip()
    
    try:
        experiments = json.loads(content)
        if isinstance(experiments, list):
            return experiments
        elif isinstance(experiments, dict):
            return [experiments]
        else:
            return []
    except Exception:
        return []

def _extract_final_law(response_text: str, function_signature: str):
    # Find the last occurrence of <final_law> and the first </final_law> after it
    last_start = response_text.rfind('<final_law>')
    if last_start == -1:
        return False, f"{function_signature} return float('nan')"
    
    last_end = response_text.find('</final_law>', last_start)
    if last_end == -1:
        return False, f"{function_signature} return float('nan')"
    
    # Extract the content between the last <final_law> and </final_law>
    final_content = response_text[last_start + len('<final_law>'):last_end].strip()
    
    # Extract the function definition using a robust pattern
    function_pattern = r'(def discovered_law.*?(?=\ndef|\Z))'
    function_match = re.findall(function_pattern, final_content, re.DOTALL)
    
    if function_match:
        return True, function_match[-1].strip()  # Get the last function match in the content
    else:
        return False, f"{function_signature} return float('nan')"

def _call_llm_and_process_response(messages: List[Dict[str, str]], model_name: str, trial_info: Dict[str, Any]) -> Tuple[List[Dict[str, str]], int, str]:
    """Calls the LLM API, processes the response, and updates the message history."""
    response_text, reasoning_response, tokens = call_llm_api(messages, model_name=model_name, trial_info=trial_info)
    
    if response_text is None:
        response_text = ""
    
    # Combine main response with reasoning if available
    if reasoning_response and reasoning_response.strip():
        combined_content = f"**Reasoning Process:**\n{reasoning_response}\n\n**Main Response:**\n{response_text}"
    else:
        combined_content = response_text
        
    messages.append({"role": "assistant", "content": combined_content})
    return messages, tokens, response_text

def conduct_exploration(module: Any, model_name: str, noise_level: float, difficulty: str = 'easy', system: str = 'vanilla_equation', law_version: str = None, max_turns: int = 10, trial_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Manages the iterative exploration process with the LLM.

    Args:
        module: The physics module (e.g., m0_gravity).
        model_name: The name of the LLM to use.
        noise_level: The noise level for experiments.
        difficulty: The difficulty level of the ground truth law ('easy', 'medium', 'hard').
        system: The experiment system ('vanilla_equation', 'simple_system', 'complex_system').
        max_turns: The maximum number of interaction rounds.
        trial_info: Optional trial information dictionary.

    Returns:
        A dictionary containing the results of the exploration.
    """
    base_prompt = BASE_PROMPT.format(max_turns=max_turns)
    if "nemotron" in model_name:
        base_prompt = "detailed thinking on \n" + base_prompt
    messages = [{"role": "system", "content": base_prompt}]
    messages.append({"role": "user", "content": module.get_task_prompt(system, noise_level=noise_level)})
    
    total_tokens = 0
    num_experiments_run = 0

    for turn in range(max_turns):
        messages, tokens, response_text = _call_llm_and_process_response(messages, model_name, trial_info)
        total_tokens += tokens

        # Check for final law submission
        is_submitted, submitted_law = _extract_final_law(response_text, module.FUNCTION_SIGNATURE)
        if is_submitted:
            return {
                "status": "completed",
                "submitted_law": submitted_law,
                "rounds": turn + 1,
                "total_tokens": total_tokens,
                "num_experiments": num_experiments_run,
                "chat_history": messages
            }

        # Check for experiment request
        experiments_to_run = parse_experiment_request(response_text if response_text is not None else "")
        
        if experiments_to_run:
            num_experiments_run += len(experiments_to_run)
            results = []
            for exp in experiments_to_run:
                # Pass system and law_version to run_experiment_for_module
                result = module.run_experiment_for_module(**exp, noise_level=noise_level, difficulty=difficulty, system=system, law_version=law_version)
                if system == "vanilla_equation":
                    result = "{:.15e}".format(result)            
                results.append(result)

            # Format results for the LLM as JSON
            output_str = f"<experiment_output>\n{json.dumps(results)}\n</experiment_output>"
            messages.append({"role": "user", "content": output_str})
        else:
            # If no valid action, prompt the LLM to act
            messages.append({"role": "user", "content": "Invalid response. Please use <run_experiment> tag with the correct JSON format or <final_law> tag to submit the law."})
    # If max turns are reached, force submission
    final_prompt = "You have used all your experiment turns. Please submit your final law now using the <final_law> tag."
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + final_prompt
    else:
        messages.append({"role": "user", "content": final_prompt})
    
    messages, tokens, response_text = _call_llm_and_process_response(messages, model_name, trial_info)
    total_tokens += tokens

    _, submitted_law = _extract_final_law(response_text, module.FUNCTION_SIGNATURE)
    return {
        "status": "max_turns_reached",
        "submitted_law": submitted_law,
        "rounds": max_turns,
        "total_tokens": total_tokens,
        "num_experiments": num_experiments_run,
        "chat_history": messages
    }
