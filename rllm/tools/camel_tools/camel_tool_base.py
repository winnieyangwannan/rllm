try:
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

import os
from typing import Any

from dotenv import load_dotenv

from rllm.tools.tool_base import Tool, ToolOutput

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


class CamelTool(Tool):
    def __init__(self, function_tool: FunctionTool):
        self.function_tool = function_tool
        super().__init__(name=function_tool.get_openai_function_schema()["name"], description=function_tool.get_openai_function_schema()["description"])

    @property
    def json(self) -> dict[str, Any]:
        return self.function_tool.get_openai_tool_schema()

    def forward(self, **kwargs) -> str:
        try:
            result = self.function_tool(**kwargs)
            return ToolOutput(name=self.name, output=str(result))
        except Exception as e:
            print(f"Error: {e}")
            return ToolOutput(name=self.name, output=str(e))
