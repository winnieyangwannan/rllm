from rllm.environments.tools.code_tools import E2BPythonInterpreter, PythonInterpreter
from rllm.environments.tools.math_tools import CalculatorTool
from rllm.environments.tools.web_tools import GoogleSearchTool, FirecrawlTool, TavilyTool


TOOL_REGISTRY = {
    'calculator': CalculatorTool,
    'e2b_python': E2BPythonInterpreter,
    'python': PythonInterpreter,
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily': TavilyTool,
}

__all__ = [
    'CalculatorTool',
    'PythonInterpreter',
    'E2BPythonInterpreter',
    'GoogleSearchTool',
    'FirecrawlTool',
    'TavilyTool',
    'TOOL_REGISTRY',
]
