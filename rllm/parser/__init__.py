from rllm.parser.tool_parser_base import ToolParser
from rllm.parser.r1_tool_parser import R1ToolParser
from rllm.parser.qwen_tool_parser import QwenToolParser
from rllm.parser.deepscaler_tool_parser import DeepScalerToolParser


PARSER_REGISTRY = {
    "deepscaler": DeepScalerToolParser,
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}



__all__ = ["R1ToolParser", "QwenToolParser", "DeepScalerToolParser", "ToolParser"]