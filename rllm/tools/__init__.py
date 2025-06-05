from rllm.tools.code_tools import (
    E2BPythonInterpreter,
    LCBPythonInterpreter,
    PythonInterpreter,
)
from rllm.tools.registry import ToolRegistry
from rllm.tools.web_tools import (
    FirecrawlTool,
    GoogleSearchTool,
    TavilyExtractTool,
    TavilySearchTool,
)

# Define default tools dict
DEFAULT_TOOLS = {
    'e2b_python': E2BPythonInterpreter,
    'local_python': PythonInterpreter,
    'python': LCBPythonInterpreter, # Make LCBPythonInterpreter the default python tool for CodeExec.
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily_extract': TavilyExtractTool,
    'tavily_search': TavilySearchTool,
}

# Create the singleton registry instance and register all default tools
tool_registry = ToolRegistry()
tool_registry.register_all(DEFAULT_TOOLS)

__all__ = [
    'PythonInterpreter',
    'E2BPythonInterpreter',
    'LCBPythonInterpreter',
    'GoogleSearchTool',
    'FirecrawlTool',
    'TavilyExtractTool',
    'TavilySearchTool',
    'ToolRegistry',
    'tool_registry',
]