from rllm.environments.tools.code_tools import E2BPythonInterpreter, PythonInterpreter, LCBPythonInterpreter
from rllm.environments.tools.math_tools import CalculatorTool
from rllm.environments.tools.web_tools import GoogleSearchTool, FirecrawlTool, TavilyTool


TOOL_REGISTRY = {
    'calculator': CalculatorTool,
    'e2b_python': E2BPythonInterpreter,
    'python': PythonInterpreter,
    'lcb_python': LCBPythonInterpreter,
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily': TavilyTool,
}

__all__ = [
    'CalculatorTool',
    'PythonInterpreter',
    'E2BPythonInterpreter',
    'LCBPythonInterpreter',
    'GoogleSearchTool',
    'FirecrawlTool',
    'TavilyTool',
    'TOOL_REGISTRY',
]
