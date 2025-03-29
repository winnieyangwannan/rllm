import json
from typing import List, Dict, Union

from rllm.tools.tool_base import ToolCall, ToolInputs, ToolOutputs, ToolOutput
from rllm.tools.code_tools.code_tool import CodeToolOutput
from rllm.parser.tool_parser_base import ToolParser

class DeepScalerToolParser(ToolParser):
    """Parser for DeepScaler tool call formats, focused on Python code blocks."""
    
    def __init__(self, model: str = None, tokenizer = None):
        """Initialize the parser.
        
        Args:
            model (str): Model name for tokenizer (optional)
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.tool_start_token = "```python"
        self.tool_end_token = "```"
        
        self.tool_response_start_token = "```output\n"
        self.tool_response_end_token = "\n```\n"
        super().__init__(model, tokenizer)
        
    def parse_input(self, model_output: str) -> ToolInputs:
        """Parse tool calls from model output.
        
        Args:
            model_output (str): Text containing tool calls
            
        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_python_tool_calls(model_output)
        
        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], parameters=tc["parameters"]) for tc in tool_calls_dicts]
        return ToolInputs(inputs=tool_calls)

    def parse_output(self, tool_outputs: Union[ToolOutput, ToolOutputs]) -> str:
        """Parse tool outputs from model output.
        
        Args:
            tool_outputs: Tool outputs to convert to string format
            
        Returns:
            str: Formatted tool outputs
        
        Example:
        ```output
        STDOUT: The circumference of a circle with radius 5 is 31.41592653589793
        ```
        """
        if isinstance(tool_outputs, ToolOutput):
            tool_outputs = ToolOutputs(outputs=[tool_outputs])
        assert isinstance(tool_outputs.outputs[0], CodeToolOutput)
        
        output_strs = []
        for output in tool_outputs.outputs:
            assert isinstance(output, CodeToolOutput), f"Tool output is not a CodeToolOutput, please use DeepScalerToolParser with code tools."
            output_str = f"{self.tool_response_start_token}\n"
            if output.error:
                output_str += f"ERROR: {output.error}\n"
            else:
                if output.stdout:
                    output_str += f"STDOUT: {output.stdout}\n"
                if output.stderr:
                    output_str += f"STDERR: {output.stderr}\n"
                if output.output:
                    output_str += f"OUTPUT: {output.output}\n"
            output_str += f"{self.tool_response_end_token}"
            output_strs.append(output_str)
        return "\n".join(output_strs)

    def parse_python_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text using Python code block format.
        
        Format:
        ```python
        some_python_code_here
        ```
        
        Returns:
            List[Dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []
        
        # Find all occurrences of ```python blocks
        while self.tool_start_token in text:
            # Find start of Python block
            start = text.find(self.tool_start_token) + len(self.tool_start_token)
            # Find end of Python block
            end = text.find(self.tool_end_token, start)
            if end == -1:
                break
                
            # Extract the Python code content
            python_code = text[start:end].strip()
            
            # Create tool call with python function and code parameter
            tool_calls.append({
                "name": "python",
                "parameters": {"code": python_code}
            })
                
            # Move to next potential Python block
            text = text[end + 3:]
        
        return tool_calls


def main():
    # Initialize the parser
    parser = DeepScalerToolParser(model='agentica-org/DeepScaleR-1.5B-Preview')
    
    # Example model output with Python code tool calls
    model_output = """
    Let me help you execute some Python code.
    
    ```python
import math

def calculate_circumference(radius):
    return 2 * math.pi * radius
    
print(f"The circumference of a circle with radius 5 is {calculate_circumference(5)}")
    ```
    
    And here's another example:
    
    ```python
import random

# Generate 5 random numbers
numbers = [random.randint(1, 100) for _ in range(5)]
print(f"Random numbers: {numbers}")
print(f"Sum of numbers: {sum(numbers)}")
    ```
    """
    
    # Parse the tool calls
    tool_inputs = parser.parse_input(model_output)
    
    # Access the parsed tool calls
    print(f"Found {len(tool_inputs.inputs)} tool calls:")
    for i, tool_call in enumerate(tool_inputs.inputs, 1):
        print(f"\nTool call {i}:")
        print(f"Name: {tool_call.name}")
        print(f"Parameters: {tool_call.parameters}")
    
    from rllm.tools import LCBPythonInterpreter
    lcb_tool = LCBPythonInterpreter()
    tool_outputs = [lcb_tool(**tool_call.parameters) for tool_call in tool_inputs.inputs]
    tool_outputs = ToolOutputs(outputs=tool_outputs)
    print(tool_outputs)

    
    # Format the tool outputs
    formatted_output = parser.parse_output(tool_outputs)
    print("\nFormatted tool outputs:")
    print(formatted_output)


if __name__ == "__main__":
    main()

