import re
import ast
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class CodeExecutorBase:
    """
    Executes Python code for physics discovery experiments.
    """
    
    def __init__(self, module_name: str, difficulty: str, system: str):
        """
        Initialize the code executor base.
        
        Args:
            module_name: Name of the physics module (e.g., 'm5_radioactive_decay')
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            system: Experiment system ('vanilla_equation', 'simple_system', 'complex_system')
        """
        self.module_name = module_name
        self.difficulty = difficulty
        self.system = system
        self.module = None
        self.ground_truth_law = None
        # Load the physics module
        self._load_physics_module()
        
    def _load_physics_module(self):
        """Load the specified physics module and get ground truth law."""
        try:
            # Import the module
            module_path = f"modules.{self.module_name}"
            self.module = __import__(module_path, fromlist=['*'])
            
            # Get the ground truth law
            if hasattr(self.module, 'get_ground_truth_law'):
                self.ground_truth_law, _ = self.module.get_ground_truth_law(self.difficulty)
            
        except ImportError as e:
            raise ImportError(f"Failed to load physics module {self.module_name}: {e}")
    

    
    def extract_python_tag(self, llm_response: str) -> Optional[str]:
        """
        Extract Python code from <python> tags in LLM response.
        
        Args:
            llm_response: The LLM's response containing potential <python> tags
            
        Returns:
            Extracted Python code or None if no tags found
        """
        start_tag = '<python>'
        end_tag = '</python>'
        
        start_index = llm_response.rfind(start_tag)
        if start_index == -1:
            return None
            
        end_index = llm_response.find(end_tag, start_index)
        if end_index == -1:
            return None

        content = llm_response[start_index + len(start_tag):end_index].strip()
        return content
    
    def validate_python_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code for safety and correctness.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse the code to check syntax
            ast.parse(code)
            
            # Check for dangerous imports or operations
            dangerous_patterns = [
                r'import\s+os',
                r'import\s+sys',
                r'import\s+subprocess',
                r'__import__',
                r'eval\(',
                r'exec\(',
                r'open\(',
                r'file\(',
                r'input\(',
                r'raw_input\(',
                r'compile\(',
                r'globals\(',
                r'locals\('
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return False, f"Code contains potentially dangerous operation: {pattern}"
            
            # Allow any valid Python code - LLM has complete freedom
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a safe environment and capture output.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing execution results or error information
        """
        try:
            # Create a new namespace for execution
            namespace = {}
            
            # Capture stdout by redirecting it
            import io
            from contextlib import redirect_stdout
            
            # Execute the code and capture output
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(code, namespace)
            
            # Get the captured output
            stdout_output = output_buffer.getvalue().strip()
            
            # Return success with output
            return {
                'success': True,
                'stdout': stdout_output,
                'code': code,
                'message': 'Code executed successfully'
            }
                
        except Exception as e:
            return {
                'success': False,
                'error_type': 'execution_error',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
    
    def process_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Process LLM response and extract/execute Python code if present.
        
        Args:
            llm_response: The LLM's response
            
        Returns:
            Dictionary containing processing results
        """
        # Extract Python code if present
        python_code = self.extract_python_tag(llm_response)
        
        if not python_code:
            return {
                'has_python_tag': False,
                'message': 'No <python> tag found in response. You can use <python> on any turn to perform tasks that you want'
            }
        
        # Validate the code
        is_valid, error_message = self.validate_python_code(python_code)
        
        if not is_valid:
            return {
                'has_python_tag': True,
                'validation_success': False,
                'error_message': error_message,
                'python_code': python_code
            }
        
        # Execute the code
        execution_result = self.execute_python_code(python_code)
        
        return {
            'has_python_tag': True,
            'validation_success': True,
            'execution_result': execution_result,
            'python_code': python_code
        }
    
    def format_execution_feedback(self, processing_result: Dict[str, Any], max_tool_calls: int = 3) -> str:
        """
        Format execution results for LLM consumption.
        
        Args:
            processing_result: Result from process_llm_response
            max_tool_calls: Maximum number of tool calls allowed per trial
            
        Returns:
            Formatted feedback string for the LLM
        """
        if not processing_result['has_python_tag']:
            return processing_result['message'] + "\n\n**Reminder:** You can use <python> on any turn to perform tasks that you want"
        
        if not processing_result['validation_success']:
            return f"""❌ **Python Code Validation Failed**

**Error:** {processing_result['error_message']}

**Your Code:**
```python
{processing_result['python_code']}
```

**Please fix the error and submit corrected Python code.**

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

**Reminder:** You can use <python> on any turn to perform tasks that you want"""
        
        # Success case - wrap output in python_output tags
        feedback = f"""<python_output>
✅ **Python Code Execution Successful!**

**Output:**
{execution_result['stdout'] if execution_result['stdout'] else 'No output produced'}

**Your Code:**
```python
{execution_result['code']}
```

**Reminder:** You may use <python> again this turn (limit: {max_tool_calls} per turn).
</python_output>"""
        
        return feedback
