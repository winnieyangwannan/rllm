import json
from typing import List, Dict, Union
from transformers import AutoTokenizer
from dataclasses import dataclass

from rllm.tools.tool_base import ToolInputs, ToolOutputs, ToolOutput


class ToolParser:
    def __init__(self,
                 model: str,
                 tokenizer: AutoTokenizer = None):
        # Must either pass in model or tokenizer
        assert model or tokenizer, "Must either pass in model or tokenizer"
        self.model = model
        self.tokenizer = tokenizer
        if self.model and not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def parse_input(self, model_output: str) -> ToolInputs:
        """Parses the model output into a ToolInputs object."""
        assert isinstance(model_output, str), "model_output must be a string"
        raise NotImplementedError("Subclasses must implement this method")
    
    def parse_output(self, tool_outputs: Union[ToolOutput, ToolOutputs]) -> str:
        """Converts ToolOutputs to a string representation of the tool outputs."""
        raise NotImplementedError("Subclasses must implement this method")
    