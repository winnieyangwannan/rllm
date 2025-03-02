import asyncio

from e2b_code_interpreter import AsyncSandbox


class PythonInterpreter:
    """A tool for executing Python code in a sandboxed environment."""

    sandboxes = []
    current_sandbox_index = 0

    def __init__(self, n_sandboxes=1):
        self.name = "python"

        self.n_sandboxes = n_sandboxes

        if len(PythonInterpreter.sandboxes) == 0:
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
                sandbox = await AsyncSandbox.create(api_key="", timeout=1200)
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
