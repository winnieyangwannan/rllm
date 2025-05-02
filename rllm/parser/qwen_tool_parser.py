import json
from typing import List, Dict, Union

from rllm.tools.tool_base import ToolCall, ToolInputs, ToolOutputs, ToolOutput
from rllm.parser.tool_parser_base import ToolParser

class QwenToolParser(ToolParser):
    
    def __init__(self, model: str = 'Qwen/Qwen2.5-7B'):
        """Initialize the parser with specified type and model.
        
        Args:
            model (str): Model name for tokenizer (optional)
            parser_type (str): Type of parser to use ('qwen' or other parsers you might add)
        """
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"
        super().__init__(model)

    def parse_input(self, model_output: str) -> ToolInputs:
        """Parse tool calls from model output.
        
        Args:
            model_output (str): Text containing tool calls
            
        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_qwen_tool_calls(model_output)
        
        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return ToolInputs(inputs=tool_calls)

    def parse_output(self, tool_outputs: Union[ToolOutput, ToolOutputs]) -> str:
        """Parse tool outputs from model output.
        
        Args:
            model_output (str): Text containing tool outputs
            
        Returns:
            ToolOutputs: Parsed tool outputs
        """
        if isinstance(tool_outputs, ToolOutput):
            tool_outputs = ToolOutputs(outputs=[tool_outputs])
        try:
            results = [
                {
                    "role": "tool",
                    "name": o.name,
                    "content": json.dumps(o.output)
                }
                for o in tool_outputs.outputs
            ]
        except json.JSONDecodeError:
            raise ValueError(f"Tool output {tool_outputs.outputs} is not a valid JSON object")
        
        return self.tokenizer.apply_chat_template(results, tokenize=False, special_tokens=False)
        
    def parse_qwen_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text using a simple token format.
    
        Format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
        
        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """

        tool_calls = []
        
        # Return empty list if no tool calls found
        if self.tool_call_begin not in text:
            return tool_calls
            
        # Process all tool calls in the text
        while self.tool_call_begin in text:
            start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = text.find(self.tool_call_end)
            if end == -1:
                break
                
            # Extract and parse the JSON content
            json_content = text[start:end].strip()
            print(f"json_content: {json_content}")
            try:
                call_data = json.loads(json_content)
                # Convert to common format matching parse_tool_calls output
                tool_calls.append({
                    "name": call_data["name"],
                    "arguments": call_data["arguments"]
                })
            except json.JSONDecodeError:
                print(f"Error parsing tool call: {json_content}")
                break
                
            # Move to next potential tool call
            text = text[end + len(self.tool_call_end):]
        
        return tool_calls
    
    def get_tool_prompt(self, tools_schema):
        return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""


def main():
    # Initialize the parser
    parser = QwenToolParser()
    
    # Example model output with tool calls
    model_output = """
    I'll help you find the weather information for New York.
    
    <tool_call>{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}</tool_call>
    
    I'll also search for restaurants in the area.
    
    <tool_call>{"name": "search_restaurants", "arguments": {"location": "New York", "cuisine": "Italian", "price_range": "moderate"}}</tool_call>
    """
    
    # Parse the tool calls
    tool_outputs = parser.parse_input(model_output)
    
    # Access the parsed tool calls
    print(f"Found {len(tool_outputs.inputs)} tool calls:")
    for i, tool_call in enumerate(tool_outputs.inputs, 1):
        print(f"\nTool call {i}:")
        print(f"Name: {tool_call.name}")
        print(f"Parameters: {tool_call.parameters}")

if __name__ == "__main__":
    main()