import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

from e2b_code_interpreter import AsyncSandbox


class E2BPythonInterpreter:
    """A tool for executing Python code in a sandboxed environment."""

    sandboxes = []
    current_sandbox_index = 0

    def __init__(self, n_sandboxes=1):
        self.name = "python"

        self.n_sandboxes = n_sandboxes

        if len(E2BPythonInterpreter.sandboxes) == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._init_sandbox())

    def __del__(self):
        """Cleanup when the interpreter is destroyed."""
        if not self.sandboxes or asyncio.get_event_loop().is_closed():
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._kill_sandbox())
            else:
                loop.run_until_complete(self._kill_sandbox())
        except Exception:
            pass

    info = {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute python code in a sandbox and return result, prefer using this to evaluate mathematical expression instead of calculating it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute in a single cell",
                    }
                },
                "required": ["code"],
            },
        },
    }

    async def _init_sandbox(self):
        """Initialize multiple sandbox environments."""
        if not self.sandboxes:
            print(f"Creating {self.n_sandboxes} sandboxes")
            for _ in range(self.n_sandboxes):
                sandbox = await AsyncSandbox.create(api_key="", timeout=3600)
                self.sandboxes.append(sandbox)

    async def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        for sandbox in self.sandboxes:
            await sandbox.kill()
        self.sandboxes = []

    async def _execute_python(self, code: str = "", **kwargs) -> str:
        """
        Execute Python code in one of the sandboxes using round-robin distribution.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        # Get next sandbox in round-robin fashion
        if "id" in kwargs:
            print("id", kwargs["id"])
            self.current_sandbox_index = kwargs["id"] % self.n_sandboxes
            print("current idx:", self.current_sandbox_index)
        else:
            self.current_sandbox_index = (
                self.current_sandbox_index + 1
            ) % self.n_sandboxes
        sandbox = self.sandboxes[self.current_sandbox_index]

        retries = 3
        while retries > 0:
            try:
                execution = await sandbox.run_code(code, timeout=20)
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    await sandbox.kill()
                    self.sandboxes[
                        self.current_sandbox_index
                    ] = await AsyncSandbox.create(api_key="", timeout=3600)
                    return "Code sandbox error, sandbox restarted. Please try again."
                await asyncio.sleep(1)
        if execution.error:
            return f"Error: {execution.error.name} - {execution.error.value}"
        if len(execution.results) == 0:
            if len(execution.logs.stdout) > 0:
                return execution.logs.stdout[0]
            else:
                print("Emtpy results:", str(execution))
                return "Empty Results"
        return execution.results[0].text

    async def execute(self, **kwargs):
        """Execute Python code in sandbox with given arguments."""
        return await self._execute_python(**kwargs)


class LocalPythonInterpreter:
    """A tool for executing Python code in a sandboxed environment."""

    def __init__(self, n_sandboxes=1):
        self.name = "python"
        self.n_workers = n_sandboxes
        self.pool = ProcessPoolExecutor(max_workers=n_sandboxes)
        
    def __del__(self):
        """Cleanup when the interpreter is destroyed."""
        if hasattr(self, 'pool'):
            self.pool.shutdown(wait=False)

    info = {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute python code in a sandbox and return result, prefer using this to evaluate mathematical expression instead of calculating it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute in a single cell",
                    }
                },
                "required": ["code"],
            },
        },
    }

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
        LocalPythonInterpreter._check_requirements()
        
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
                return f"Error: Execution timed out after {timeout} seconds"
            except Exception as e:
                return f"Error: {type(e).__name__} - {str(e)}"
            finally:
                os.unlink(f.name)

    async def _execute_python(self, code: str = "", timeout: int = 10, **kwargs) -> str:
        """
        Execute Python code in a sandboxed environment using the process pool.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds before timing out
            
        Returns:
            Execution result as string
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.pool, 
                self._execute_in_subprocess,
                code,
                timeout
            )
            return result
        except Exception as e:
            return f"Error: {type(e).__name__} - {str(e)}"

    async def execute(self, **kwargs):
        """Execute Python code in sandbox with given arguments."""
        return await self._execute_python(**kwargs)


class PythonInterpreter:
    def __new__(cls, type: str = "local", n_sandboxes: int = 1):
        """
        Create a Python interpreter of the specified type.
        
        Args:
            interpreter_type: Type of interpreter ('local' or 'e2b')
            n_sandboxes: Number of sandboxes/workers to create
            
        Returns:
            An instance of either LocalPythonInterpreter or E2BPythonInterpreter
        """
        interpreter_type = type.lower()
        if interpreter_type == "local":
            return LocalPythonInterpreter(n_sandboxes=n_sandboxes)
        elif interpreter_type == "e2b":
            return E2BPythonInterpreter(n_sandboxes=n_sandboxes)
        else:
            raise ValueError(f"Unknown interpreter type: {interpreter_type}. Choose 'local' or 'e2b'.") 
        


if __name__ == "__main__":
    # Create a local Python interpreter instance
    interpreter = PythonInterpreter(type="local")

    # Example code to execute
    test_code = """
print('Hello from Python interpreter!')
x = 5
y = 10
print(f'Sum of {x} and {y} is: {x + y}')
"""

    # Run the code using asyncio
    async def test_interpreter():
        result = await interpreter.execute(code=test_code)
        print("Execution result:")
        print(result)

    # Run the async test
    asyncio.run(test_interpreter())