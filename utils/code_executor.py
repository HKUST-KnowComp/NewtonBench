import re
import ast
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from .code_executor_base import CodeExecutorBase


class CodeExecutor(CodeExecutorBase):
    """
    Executes Python code for physics discovery experiments
    with per-turn Python call limits instead of per-trial limits.
    """
    
    def __init__(self, module_name: str, difficulty: str, system: str):
        """
        Initialize the code executor.
        
        Args:
            module_name: Name of the physics module (e.g., 'm5_radioactive_decay')
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        """
        super().__init__(module_name, difficulty, system)
        self.python_calls_this_turn = 0
        self.turn_number = 0
        self.max_python_calls_per_turn = 1
        
    def reset_turn_counter(self):
        """Reset Python call counter for new turn."""
        self.python_calls_this_turn = 0
        self.turn_number += 1
        print(f"[Code Assisted] Turn {self.turn_number} started - Python calls reset to 0/{self.max_python_calls_per_turn}")
    
    def can_execute_python(self) -> bool:
        """Check if more Python calls allowed this turn."""
        return self.python_calls_this_turn < self.max_python_calls_per_turn
    
    def increment_python_counter(self):
        """Increment Python call counter for current turn."""
        self.python_calls_this_turn += 1
        remaining = self.max_python_calls_per_turn - self.python_calls_this_turn
        print(f"[Code Assisted] Turn {self.turn_number}: Python call {self.python_calls_this_turn}/{self.max_python_calls_per_turn} used ({remaining} remaining)")
    
    def get_turn_usage_info(self) -> str:
        """Get formatted usage information for current turn."""
        return f"Turn {self.turn_number}: {self.python_calls_this_turn}/{self.max_python_calls_per_turn} Python calls used this turn"
    
    def process_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Process LLM response and extract/execute Python code if present.
        Enhanced with per-turn limit checking.
        
        Args:
            llm_response: The LLM's response
            
        Returns:
            Dictionary containing processing results with turn-specific info
        """
        # Extract Python code if present
        python_code = self.extract_python_tag(llm_response)
        
        if not python_code:
            return {
                'has_python_tag': False,
                'message': 'No <python> tag found in response. You can use <python> on any turn to perform tasks that you want',
                'turn_usage_info': self.get_turn_usage_info(),
                'can_execute_more': self.can_execute_python()
            }
        
        # Check if we can execute more Python this turn
        if not self.can_execute_python():
            return {
                'has_python_tag': True,
                'limit_reached': True,
                'message': f'Python call limit reached for turn {self.turn_number}. You have used {self.max_python_calls_per_turn} Python calls this turn.',
                'turn_usage_info': self.get_turn_usage_info(),
                'can_execute_more': False,
                'python_code': python_code
            }
        
        # Validate the code
        is_valid, error_message = self.validate_python_code(python_code)
        
        if not is_valid:
            return {
                'has_python_tag': True,
                'validation_success': False,
                'error_message': error_message,
                'python_code': python_code,
                'turn_usage_info': self.get_turn_usage_info(),
                'can_execute_more': self.can_execute_python()
            }
        
        # Execute the code
        execution_result = self.execute_python_code(python_code)
        
        # Increment counter after successful execution
        self.increment_python_counter()
        
        return {
            'has_python_tag': True,
            'validation_success': True,
            'execution_result': execution_result,
            'python_code': python_code,
            'turn_usage_info': self.get_turn_usage_info(),
            'can_execute_more': self.can_execute_python()
        }
    
    def format_execution_feedback(self, processing_result: Dict[str, Any]) -> str:
        """
        Format execution results for LLM consumption with turn-specific information.
        
        Args:
            processing_result: Result from process_llm_response
            
        Returns:
            Formatted feedback string for the LLM
        """
        if not processing_result['has_python_tag']:
            return f"""{processing_result['message']}

**{processing_result['turn_usage_info']}**
**Reminder:** You can use <python> on any turn to perform tasks that you want"""
        
        if processing_result.get('limit_reached', False):
            return f"""⚠️ **Python Call Limit Reached for This Turn**

{processing_result['message']}

**Your Code:**
```python
{processing_result['python_code']}
```

**Next Steps:**
- Continue with your analysis using the information you have gathered
- Submit experiments using <run_experiment> tags if needed
- Submit your final law using <final_law> tags when ready
- Your Python call limit will reset to {self.max_python_calls_per_turn} in the next turn

**{processing_result['turn_usage_info']}**"""
        
        if not processing_result['validation_success']:
            return f"""❌ **Python Code Validation Failed**

**Error:** {processing_result['error_message']}

**Your Code:**
```python
{processing_result['python_code']}
```

**Please fix the error and submit corrected Python code.**

**{processing_result['turn_usage_info']}**
**Reminder:** You can use <python> on any turn to perform tasks that you want"""
        
        execution_result = processing_result['execution_result']
        
        if not execution_result['success']:
            return f"""❌ **Python Code Execution Failed**

**Error Type:** {execution_result['error_type']}
**Error Message:** {execution_result['error_message']}

**Your Code:**
```python
{processing_result['python_code']}
```

**Please fix the error and submit corrected Python code.**

**{processing_result['turn_usage_info']}**
**Reminder:** You can use <python> on any turn to perform tasks that you want"""
        
        # Success case - wrap output in python_output tags with turn info
        remaining_calls = self.max_python_calls_per_turn - self.python_calls_this_turn
        feedback = f"""<python_output>
✅ **Python Code Execution Successful!**

**Output:**
{execution_result['stdout'] if execution_result['stdout'] else 'No output produced'}

**Your Code:**
```python
{execution_result['code']}
```

**{processing_result['turn_usage_info']}**
{f'**Reminder:** You may use <python> again this turn ({remaining_calls} calls remaining).' if remaining_calls > 0 else f'**Turn Limit Reached:** You have used all {self.max_python_calls_per_turn} Python calls for this turn.'}
</python_output>"""
        
        return feedback
