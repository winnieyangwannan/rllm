from rllm.environments.tools.code_tools.local_tool import PythonInterpreter
from rllm.environments.tools.math_tools.calculator import CalculatorTool
from rllm.environments.tools.web_tools import GoogleSearchTool, FirecrawlTool, TavilyTool

TOOL_REGISTRY = {
    'calculator': CalculatorTool,
    'python_interpreter': PythonInterpreter,
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily': TavilyTool,
}