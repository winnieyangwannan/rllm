from rllm.tools.code_tools import E2BPythonInterpreter, PythonInterpreter, LCBPythonInterpreter
from rllm.tools.math_tools import CalculatorTool
from rllm.tools.web_tools import GoogleSearchTool, FirecrawlTool, TavilyTool

from rllm.tools.camel_tools.audio_tools import Audio2TextCamel, AskQuestionAboutAudioCamel
from rllm.tools.camel_tools.excel_tools import ExtractExcelContentCamel
from rllm.tools.camel_tools.image_analysis_tools import ImageToTextCamel, AskQuestionAboutImageCamel
from rllm.tools.camel_tools.search_tools import SearchWikiCamel, SearchGoogleCamel
from rllm.tools.camel_tools.video_analysis_tools import AskQuestionAboutVideoCamel
from rllm.tools.camel_tools.browser_tools import BrowseUrlCamel
from rllm.tools.camel_tools.code_execution_tools import ExecuteCodeCamel
from rllm.tools.camel_tools.document_toolkit import DocumentProcessingCamel
TOOL_REGISTRY = {
    'calculator': CalculatorTool,
    'e2b_python': E2BPythonInterpreter,
    'local_python': PythonInterpreter,
    'python': LCBPythonInterpreter, # Make LCBPythonInterpreter the default python tool for CodeExec.
    'google_search': GoogleSearchTool,
    'firecrawl': FirecrawlTool,
    'tavily': TavilyTool,
    'audio2text': Audio2TextCamel,
    'ask_question_about_audio': AskQuestionAboutAudioCamel,
    'extract_excel_content': ExtractExcelContentCamel,
    'image_to_text': ImageToTextCamel,
    'ask_question_about_image': AskQuestionAboutImageCamel,
    'search_wiki': SearchWikiCamel,
    'search_google': SearchGoogleCamel,
    'ask_video': AskQuestionAboutVideoCamel,
    'browse_url': BrowseUrlCamel,
    'execute_code': ExecuteCodeCamel,
    'extract_document_content': DocumentProcessingCamel,
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
    # Audio tools
    'Audio2TextCamel',
    'AskQuestionAboutAudioCamel',
    # Excel tools
    'ExtractExcelContentCamel',
    # Image tools
    'ImageToTextCamel',
    'AskQuestionAboutImageCamel',
    # Search tools
    'SearchWikiCamel',
    'SearchGoogleCamel',
    # Video tools
    'AskQuestionAboutVideoCamel',
    # Browser tools
    'BrowseUrlCamel',
    # Code execution tools
    'ExecuteCodeCamel',
]
