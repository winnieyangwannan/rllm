import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict

from rllm.environments.tools.tool_base import Tool


class PythonInterpreter(Tool):
    """A tool for executing Python code in a sandboxed environment."""

    def __init__(self, n_sandboxes=1):
        self.n_workers = n_sandboxes
        self.pool = ProcessPoolExecutor(max_workers=n_sandboxes)
        super().__init__(
            name="local_python_sandbox_tool",
            description="Execute python code in a local sandbox environment. Returns results and standard output/error."
        )

    @property
    def json(self) -> Dict[str, Any]:
        """Return the tool's information in the required format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Local sandbox to execute the python code in a single cell",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds before timing out",
                            "default": 12
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    def forward(self, code: str, timeout: int = 12) -> str:
        """
        Synchronous implementation of Python code execution in a sandbox.
        Uses the process pool for isolation but blocks until completion.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Execution result as string
        """
        try:
            # Submit the job to the process pool and wait for its result
            future = self.pool.submit(self._execute_in_subprocess, code, timeout)
            return future.result(timeout=timeout)
        except Exception as e:
            return f"Error: {type(e).__name__} - {str(e)}"

    @staticmethod
    def _check_requirements():
        """Check if required packages are installed and install if missing."""
        required_packages = {
            'sympy': 'sympy',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib'
        }
        
        missing_packages = []
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            try:
                import subprocess
                import sys
                
                # Install missing packages using pip
                subprocess.check_call([
                    sys.executable, 
                    '-m', 'pip', 
                    'install', 
                    '--quiet',
                    *missing_packages
                ])
                print(f"Successfully installed: {', '.join(missing_packages)}")
            except Exception as e:
                raise RuntimeError(f"Failed to install required packages: {str(e)}")

    @staticmethod
    def _execute_in_subprocess(code: str, timeout: int = 10) -> str:
        """Execute code in a separate process with resource limits."""
        # First check and install requirements
        PythonInterpreter._check_requirements()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code to capture stdout and stderr, and last expression value
            wrapped_code = f"""
import sys
import io
import contextlib
import math

def _format_value(val):
    if val is None:
        return ''
    return repr(val)

output = io.StringIO()
with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    try:
        # Split code into lines and get the last line
        code_lines = {repr(code)}.rstrip().split('\\n')
        # Execute all lines except the last one
        if len(code_lines) > 1:
            exec('\\n'.join(code_lines[:-1]))
        # For the last line, try eval first, if it fails, use exec
        try:
            last_value = eval(code_lines[-1])
            print(_format_value(last_value))
        except SyntaxError:
            exec(code_lines[-1])
    except Exception as e:
        print(f"Error: {{type(e).__name__}} - {{str(e)}}")

print(output.getvalue())
"""
            f.write(wrapped_code)
            f.flush()
            
            try:
                # Execute with resource limits
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,  # Use provided timeout
                )
                output = result.stdout.strip() or result.stderr.strip() or "Empty Results"
                return output
            except subprocess.TimeoutExpired:
                return {
                    "error": f"Execution timed out after {timeout} seconds"
                } 
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return {
                    "error": f"{type(e).__name__} - {str(e)}\n{traceback.format_exc()}"
                }
            finally:
                os.unlink(f.name)

    def __del__(self):
        """Cleanup when the interpreter is destroyed."""
        try:
            if hasattr(self, 'pool'):
                self.pool.shutdown(wait=False)
        except Exception:
            pass


if __name__ == "__main__":
    # Create a Python interpreter instance
    interpreter = PythonInterpreter(n_sandboxes=1)

    # Example code to execute
    test_code = """
print('Hello from Python interpreter!')
x = 5
y = 10
print(f'Sum of {x} and {y} is: {x + y}')
math.hello
"""

    # Run code synchronously
    print("Synchronous result:")
    print(interpreter(code=test_code, use_async=False))
    
    # Run the code using asyncio
    # async def test_async_interpreter():
    #     result = await interpreter(code=test_code, use_async=True)
    #     print("\nAsynchronous result:")
    #     print(result)

    # # Run the async test
    # asyncio.run(test_async_interpreter())
