try:
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from typing import Any, Dict
from rllm.tools.tool_base import Tool, ToolOutput
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

class CamelTool(Tool):
    def __init__(self, function_tool: FunctionTool):
        self.function_tool = function_tool
        super().__init__(
            name=function_tool.get_openai_function_schema()['name'],
            description=function_tool.get_openai_function_schema()['description']
        )
    
    @property
    def json(self) -> Dict[str, Any]:
        return self.function_tool.get_openai_tool_schema()
    
    def forward(self, **kwargs) -> str:
        return ToolOutput(name=self.name, output=self.function_tool(**kwargs))