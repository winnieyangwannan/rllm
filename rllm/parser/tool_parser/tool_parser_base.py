from abc import ABC, abstractmethod
from typing import List

from rllm.tools.tool_base import ToolCall


class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> List[ToolCall]:
        """Extract tool calls from the model response."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method")