
from e2b_code_interpreter import AsyncSandbox

class PythonInterpreter:
    """A tool for executing Python code in a sandboxed environment."""
    sandbox = None

    def __init__(self):
        self.name = "python_interpreter"
        
    @property
    def info(self):
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": "Execute python code in a sandbox and return result, prefer using this to evaluate mathematical expression instead of calculating it yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute in a single cell"
                        }
                    },
                    "required": ["code"]
                }
            }
        }

    async def execute(self, **kwargs):
        """Execute Python code in sandbox with given arguments."""
        if PythonInterpreter.sandbox is None:
            await self._init_sandbox()
        return await self._execute_python(**kwargs)

    async def _init_sandbox(self):
        """Initialize the sandbox environment."""
        PythonInterpreter.sandbox = await AsyncSandbox.create(
            api_key=""
        )  # need an API key here for e2b sandbox

    async def _kill_sandbox(self):
        """Clean up sandbox resources."""
        if PythonInterpreter.sandbox is not None:
            await PythonInterpreter.sandbox.kill()
            PythonInterpreter.sandbox = None

    async def _execute_python(self, code: str = "", **kwargs) -> str:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result as string
        """
        print("Execute SANDBOX")
        execution = await PythonInterpreter.sandbox.run_code(code)
        print(execution)
        return str(execution)