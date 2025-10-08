import json
import time
from typing import Dict, List, Any, Optional
from .code_executor import CodeExecutor
from .call_llm_api import call_llm_api
import re


def conduct_code_assisted_exploration(
    module,
    model_name: str,
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: str = None,
    trial_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Conduct physics discovery exploration using code assistant with per-turn Python call limits.
    
    Args:
        module: The physics module to explore
        model_name: Name of the LLM model to use
        noise_level: Noise level for experiments
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        law_version: Specific law version to use
        trial_info: Additional trial information
        
    Returns:
        Dictionary containing exploration results
    """
    
    # Initialize trial_info if not provided
    trial_info = trial_info or {}
    
    # Initialize code executor
    code_executor = CodeExecutor(
        module_name=module.__name__.split('.')[-1],
        difficulty=difficulty,
        system=system
    )

    max_turns = 10  # Limit to prevent infinite loops
    # Create code assisted agent-specific system prompt
    system_prompt = create_code_assisted_system_prompt(module, difficulty, system, max_turns)
    
    # Initialize conversation with just the system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add the task prompt as the first user message
    task_prompt = module.get_task_prompt(system, is_code_assisted=True, noise_level=noise_level)
    messages.append({"role": "user", "content": task_prompt})
    
    # Initialize chat history with the system and task prompts
    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]
    total_tokens = 0
    num_experiments = 0
    python_tags_used_total = 0  # Counter for Python tags across entire trial
    
    # Extract trial identifier for logging
    trial_id = trial_info.get('trial_id', 'unknown') if trial_info else 'unknown'
    
    print(f"[Code Assisted Trial {trial_id}] Starting exploration for {module.__name__} - {difficulty} {system}")
    
    trial_completed = False
    for turn in range(max_turns):
        # Check if trial is already completed (final law submitted)
        if trial_completed:
            break
            
        # Reset Python call counter for new turn
        code_executor.reset_turn_counter()
        
        # Inner loop for multiple Python calls per turn
        turn_completed = False
        
        while not turn_completed and code_executor.can_execute_python():
            try:
                # Add per-turn reminder about <python> availability
                turn_reminder = f"You can use either <run_experiment> to collect more data or use <python> tag to do some analysis. You should distribute your action wisely. Only submit your law using the <final_law> tag when you are confident."
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + turn_reminder
                    chat_history[-1]["content"] += "\n\n" + turn_reminder
                else:
                    messages.append({"role": "user", "content": turn_reminder})
                    chat_history.append({"role": "user", "content": turn_reminder})
                
                api_result = call_llm_api(messages, model_name, trial_info=trial_info)
                # Handle different return formats from call_llm_api
                if len(api_result) == 3:
                    response, reasoning_content, tokens = api_result
                elif len(api_result) == 2:
                    response, tokens = api_result
                    reasoning_content = None
                else:
                    raise ValueError(f"Unexpected return format from call_llm_api: {len(api_result)} values")
                
                total_tokens += tokens

                if response is None:
                    response = ""
                
                # Combine main response with reasoning if available
                if reasoning_content and reasoning_content.strip():
                    combined_content = f"**Reasoning Process:**\n{reasoning_content}\n\n**Main Response:**\n{response}"
                else:
                    combined_content = response

                # Add LLM response to chat history
                chat_history.append({
                    "role": "assistant",
                    "content": combined_content
                })
                
                # Check if LLM has submitted final law
                if response and "<final_law>" in response and "</final_law>" in response:
                    print(f"[Code Assisted Trial {trial_id}] Final law submitted on turn {turn + 1}")
                    trial_completed = True
                    turn_completed = True
                    break

                python_pos = response.rfind('<python>')
                experiment_pos = response.rfind('<run_experiment>')
     
                # Process Python code if present
                code_result = code_executor.process_llm_response(response)
                
                if code_result['has_python_tag'] and python_pos > experiment_pos:
                    # Check if we've reached the turn limit
                    if code_result.get('limit_reached', False):
                        # Python limit reached for this turn
                        print(f"[Code Assisted Trial {trial_id}] Python call limit reached for turn {turn + 1}")
                        
                        # Format feedback for LLM with turn limit info
                        feedback = code_executor.format_execution_feedback(code_result)
                        
                        # Add feedback to conversation
                        messages.append({"role": "assistant", "content": combined_content})
                        messages.append({"role": "user", "content": feedback})
                        
                        # Add feedback to chat history
                        chat_history.append({
                            "role": "user",
                            "content": f"[Code Execution Feedback - Turn Limit]\n{feedback}"
                        })
                        
                        # End this turn
                        turn_completed = True
                        break
                    
                    # Increment Python tag counter
                    python_tags_used_total += 1
                    # Process Python code execution
                    print(f"[Code Assisted Trial {trial_id}] Processing Python code in turn {turn + 1}")
                    
                    # Format feedback for LLM with turn-specific usage info
                    feedback = code_executor.format_execution_feedback(code_result)
                    
                    # Add feedback to conversation
                    messages.append({"role": "assistant", "content": combined_content})
                    messages.append({"role": "user", "content": feedback})
                    
                    # Add feedback to chat history
                    chat_history.append({
                        "role": "user",
                        "content": f"[Code Execution Feedback]\n{feedback}"
                    })
                    
                else:
                    # No Python code or experiment tag, handle other actions
                    messages.append({"role": "assistant", "content": combined_content})
                    
                    # Check if we need to run experiments
                    if response and "<run_experiment>" in response and "</run_experiment>" in response:
                        print(f"[Code Assisted Trial {trial_id}] Running experiment {num_experiments} in turn {turn + 1}")
                        
                        # Extract experiment parameters and run experiment
                        experiment_result = run_experiment_from_response(module, response, system, noise_level, difficulty, law_version)
                        
                        if experiment_result:
                            # Add experiment results to conversation
                            if isinstance(experiment_result, list):
                                num_experiments += len(experiment_result)
                            else:
                                num_experiments += 1

                            experiment_message = format_experiment_results(experiment_result)
                            if messages and messages[-1]["role"] == "user":
                                messages[-1]["content"] += "\n\n" + experiment_message
                                chat_history[-1]["content"] += "\n\n" + f"[Experiment Results]\n{experiment_message}"
                            else:
                                messages.append({"role": "user", "content": experiment_message})
                                chat_history.append({
                                    "role": "user",
                                    "content": f"[Experiment Results]\n{experiment_message}"
                                })
                        else:
                            print(f"[Code Assisted Trial {trial_id}] Experiment {num_experiments} failed to execute")
                    else:
                        reminder_message = f"**Action Reminder:** Please use exactly 1 action per turn with correct format: <run_experiment> tag with the correct JSON format, <python> tag to execute Python code, or <final_law> tag to submit the law. {code_executor.get_turn_usage_info()}"
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] += "\n\n" + reminder_message
                            chat_history[-1]["content"] += "\n\n" + reminder_message
                        else:
                            messages.append({"role": "user", "content": reminder_message})
                            chat_history.append({"role": "user", "content": reminder_message})
                        print(f"[Code Assisted Trial {trial_id}] Invalid response in turn {turn + 1}")
                    
                    # End this turn after handling non-Python actions
                    turn_completed = True
                    break
                
                # Add small delay to prevent API rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"[Code Assisted Trial {trial_id}] Error in turn {turn + 1}: {e}")
                raise e
        
        # If we've processed all Python calls for this turn and haven't completed, continue to next turn
        if not turn_completed and not code_executor.can_execute_python():
            print(f"[Code Assisted Trial {trial_id}] Turn {turn + 1} completed - Python call limit reached")
            turn_completed = True
    
    else:
        # If max turns are reached and trial not completed, force submission
        if not trial_completed:
            final_prompt = f"**IMPORTANT:**\nYou have used all your experiment turns. Please submit your final law now using the <final_law> tag. Besides, remember that the function signature should be {module.FUNCTION_SIGNATURE}. All other variables needed to be defined inside the function. No comments are allowed inside the function."
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + final_prompt
                chat_history[-1]["content"] += "\n\n" + final_prompt
            else:
                messages.append({"role": "user", "content": final_prompt})
                chat_history.append({"role": "user", "content": final_prompt})

            # Call LLM one last time to get the final law
            api_result = call_llm_api(messages, model_name, trial_info=trial_info)
            if len(api_result) == 3:
                response, reasoning_content, tokens = api_result
            elif len(api_result) == 2:
                response, tokens = api_result
                reasoning_content = None
            else:
                raise ValueError(f"Unexpected return format from call_llm_api: {len(api_result)} values")

            total_tokens += tokens

            if reasoning_content and reasoning_content.strip():
                combined_content = f"**Reasoning Process:**\n{reasoning_content}\n\n**Main Response:**\n{response}"
            else:
                combined_content = response

            chat_history.append({"role": "assistant", "content": combined_content})

    print(f"[Code Assisted Trial {trial_id}] Exploration completed after {turn + 1} turns")
    
    # Extract final law if present
    submitted_law = extract_final_law(chat_history, module)
    
    return {
        "status": "completed" if trial_completed else "max_turns_reached",
        "submitted_law": submitted_law,
        "chat_history": chat_history,
        "rounds": turn + 1,
        "total_tokens": total_tokens,
        "python_tags_used_total": python_tags_used_total,
        "num_experiments": num_experiments,
        "exploration_mode": "code_assisted_agent"
    }

def create_code_assisted_system_prompt(module, difficulty: str, system: str, max_turns: int) -> str:
    """
    Create Code Assisted-specific system prompt.
    
    Args:
        module: The physics module
        difficulty: Difficulty level
        system: Experiment system
        
    Returns:
        System prompt string
    """
    
    # Create system prompt with per-turn Python call limits
    code_assisted_system_prompt = f"""You are an AI research assistant tasked with discovering scientific laws in a simulated universe.
Your goal is to propose experiments, analyze the data they return, and ultimately deduce the underlying scientific law.
Please note that the laws of physics in this universe may differ from those in our own.
You can perform experiments to gather data but you must follow the protocol strictly.

**Rules**:
1. **Math calculation**:
    - You are always encouraged to use the <python> tag to assist with any non-trivial mathematical reasoning. This includes, but is not limited to:
        - Performing exponentiation, logarithmic transformations, and other advanced math operations.
        - Comparing predicted outputs from your proposed law against actual experiment results.
        - Calculating metrics such as mean squared error to evaluate the accuracy of your hypotheses.
        - Performing sensitivity analysis and mathematical modeling to understand how variations in experimental conditions affect outcomes.
2. **Enhanced Tool Use**:
    - Avoid Redundant Calls: Do not call the same tool with identical parameters more than once.
    - Evaluate Before Repeating: Always review the tool's output before deciding to call another tool. Only proceed if the result is incomplete, unclear, or unsatisfactory.
    - **Turn-Based Strategy**: Use your Python calls strategically within each turn for iterative analysis and refinement.

**Workflow:**
1.  Analyze the mission description provided.
2.  Design a set of experiments to test your hypotheses.
3.  Use the `<run_experiment>` tag to submit your experimental inputs.
4.  The system will return the results (at most 20 sets of data per experiment) in an `<experiment_output>` tag.
5.  You can run up to {max_turns} rounds. Use them wisely so that before submitting your final law, ensure you have:
    - fully explored the experimental space
    - Verified your hypotheses against the data
    - made the most of the available rounds to strengthen your conclusions
6.  Use the `<python>` tags to test your hypotheses, perform calculations, or explore data between experiments.
7.  The system will return the results from the `<python>` tags in a `<python_output>` tag.
8.  **CRITICAL: Only one action per turn**:
    - `<run_experiment>` tag for running experiments
    - `<python>` tag for calculations/analysis (up to 1 call per turn)
    - `<final_law>` tag for submitting your final discovered law
9.  **NO MIXING**: Never use multiple action types in the same turn (e.g., Python + Experiment)
10. **NO DUPLICATES**: Never use multiple tags of the same type in one turn
11. After submitting <run_experiment>, wait for <experiment_output> before proceeding.
12. After submitting <python>, wait for <python_output> before proceeding.
13. You should verify your hypotheses by checking if the output from the experiments matches the output from your hypotheses.
14. You should take advantage of <python> tags to do all the tasks you deem necessary within each turn.
15. Analyze the results from `<python_output>` tags to refine your understanding.
16. When confident, submit your final discovered law using the `<final_law>` tag. This ends the mission.

**Important Notes:**
- You are equipped with one tool: <python>. This tool is only for performing complex math calculations (e.g., exponentiation, logarithms, data analysis, hypothesis testing).
- **NEVER** use the `run_python_code` tool to run experiments or submit final laws. These actions must be done using the `<run_experiment>` and `<final_law>` tags respectively.
- **NEVER** include any comments inside the submitted final laws
- Always respond with the appropriate tag when submitting experiments or final laws. The environment will handle execution and feedback."""

    return code_assisted_system_prompt

def run_experiment_from_response(module, response: str, system: str, noise_level: float, difficulty: str, law_version: str = None) -> Optional[Dict[str, Any]]:
    """
    Extract experiment parameters from LLM response and run experiment.
    
    Args:
        module: The physics module
        response: LLM response containing experiment request
        system: Experiment system
        noise_level: Noise level for experiments
        difficulty: Difficulty level
        law_version: Law version to use
        
    Returns:
        Experiment results or None if extraction failed
    """
    try:
        # Parse the <run_experiment> block from the response
        import re
        import json
        
        # Extract content between <run_experiment> tags
        start_tag = '<run_experiment>'
        end_tag = '</run_experiment>'

        if response is None:
            return None
        
        start_index = response.rfind(start_tag)
        if start_index == -1:
            return None
            
        end_index = response.find(end_tag, start_index)
        if end_index == -1:
            return None
                    
        try:
            # Parse the JSON content
            content = response[start_index + len(start_tag):end_index].strip()
            exp_data = json.loads(content)
            
            # Handle both single experiment and array of experiments
            if isinstance(exp_data, list):
                # Multiple experiments
                results = []
                for exp in exp_data:
                    if hasattr(module, 'run_experiment_for_module'):
                        result = module.run_experiment_for_module(
                            **exp, 
                            noise_level=noise_level, 
                            difficulty=difficulty,
                            system=system,
                            law_version=law_version
                        )
                        results.append(result)
                    else:
                        results.append(None)
                return results
            else:
                # Single experiment
                if hasattr(module, 'run_experiment_for_module'):
                    result = module.run_experiment_for_module(
                        **exp_data, 
                        noise_level=noise_level, 
                        difficulty=difficulty,
                        system=system,
                        law_version=law_version
                    )
                    return result
                else:
                    return None
                    
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        return None

def format_experiment_results(experiment_result: Any) -> str:
    """
    Format experiment results for LLM consumption.
    
    Args:
        experiment_result: Experiment results (can be single result, list, or None)
        
    Returns:
        Formatted results string
    """
    if experiment_result is None:
        return """[Experiment Results - Error]

Failed to run experiment. Please check your parameter format.

**Reminder:** You can use <python> tags to analyze these results. Example: <python>print("Check your experiment parameters")</python>"""
    
    # Handle list of results (multiple experiments)
    if isinstance(experiment_result, list):
        if all(r is None for r in experiment_result):
            return """[Experiment Results - Error]

All experiments failed to run. Please check your parameter format.

**Reminder:** You can use <python> tags to analyze these results. Example: <python>print("Check your experiment parameters")</python>"""
        
        # Format multiple results as <experiment_output> tags
        import json
        results_str = f"<experiment_output>\n{json.dumps(experiment_result)}\n</experiment_output>"
        return results_str + "\n\n**Reminder:** You can use <python> tags to analyze these results. Example: <python>import json; data = " + json.dumps(experiment_result[0] if experiment_result else {}) + "; print(f\"First result: {data}\")</python>"
    
    # Handle single result
    else:
        # Format single result as <experiment_output> tag
        import json
        result_str = f"<experiment_output>\n{json.dumps(experiment_result)}\n</experiment_output>"
        return result_str + "\n\n**Reminder:** You can use <python> tags to analyze these results. Example: <python>import json; data = " + json.dumps(experiment_result) + "; print(f\"Result: {data}\")</python>"

def extract_final_law(chat_history: List[Dict[str, str]], module) -> str:
    """
    Extract the final submitted law from chat history.
    
    Args:
        chat_history: List of chat messages
        module: The physics module to explore
        
    Returns:
        Extracted law string or default
    """
    # Look for final law in the last few messages
    for message in reversed(chat_history):
        if message.get('role') == 'assistant' and '<final_law>' in message.get('content', ''):
            content = message['content']
            # Extract content between <final_law> tags
            # Find the last occurrence of <final_law> and the first </final_law> after it
            last_start = content.rfind('<final_law>')
            if last_start == -1:
                continue

            last_end = content.find('</final_law>', last_start)
            if last_end == -1:
                continue

            # Extract the content between the tags
            final_content = content[last_start + len('<final_law>'):last_end].strip()

            # Extract the function definition using a robust pattern
            function_pattern = r'(def discovered_law.*?(?=\ndef|\Z))'
            function_match = re.findall(function_pattern, final_content, re.DOTALL)
            if function_match:
                return function_match[-1].strip()  # Get the last function match in the content
    
    return f"{module.FUNCTION_SIGNATURE} return float('nan')"
