import json
from typing import List, Dict, Union

from rllm.tools.tool_base import ToolCall, ToolInputs, ToolOutputs, ToolOutput
from rllm.tools.parser.tool_parser_base import ToolParser

class R1ToolParser(ToolParser):
    """Parser for R1 tool call format."""
    
    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1", tokenizer = None):
        """Initialize the R1 tool parser.
        
        Args:
            model (str): Model name for tokenizer (optional)
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.tool_calls_begin = "<｜tool▁call▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        super().__init__(model, tokenizer)

    def parse_input(self, model_output: str) -> ToolInputs:
        """Parse tool calls from model output.
        
        Args:
            model_output (str): Text containing tool calls
            
        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_r1_tool_calls(model_output)
        
        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], parameters=tc["parameters"]) for tc in tool_calls_dicts]
        return ToolInputs(inputs=tool_calls)

    def parse_output(self, tool_outputs: Union[ToolOutput, ToolOutputs]) -> str:
        """Parse tool outputs from model output.
        
        Args:
            tool_outputs: Tool outputs to convert to string format
            
        Returns:
            str: Formatted tool outputs
        """
        if isinstance(tool_outputs, ToolOutput):
            tool_outputs = ToolOutputs(outputs=[tool_outputs])
        
        results = []
        for output in tool_outputs.outputs:
            try:
                output_content = output.output
                if isinstance(output_content, (dict, list)):
                    output_content = json.dumps(output_content)
                
                # Format for R1 tool output
                results.append({
                    "role": "tool",
                    "name": output.name,
                    "content": output_content
                })
            except json.JSONDecodeError:
                raise ValueError(f"Tool output {output.output} is not a valid JSON object")
        
        return self.tokenizer.apply_chat_template(results, tokenize=False, special_tokens=False)

    def parse_r1_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text using the R1 special token format.

        Format:
        <｜tool▁calls▁begin｜> 
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
        ```json
        {"param": "value"}
        ```
        <｜tool▁call▁end｜>
        // Additional tool calls follow the same format
        <｜tool▁calls▁end｜>
        
        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []
        
        # Look for individual tool calls
        call_idx = 0
        while True:
            # Find the next tool call beginning
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break
                
            # Find the end of this tool call
            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break
                
            # Extract the content of this tool call
            call_content = text[call_start:call_end].strip()
            
            # Parse function name
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)
            
            if func_start != -1:
                # Extract function name after the prefix up to the next newline
                func_name_start = func_start + len(func_prefix)
                func_name_end = call_content.find('\n', func_name_start)
                
                if func_name_end == -1:
                    function_name = call_content[func_name_start:].strip()
                else:
                    function_name = call_content[func_name_start:func_name_end].strip()
            else:
                # If function prefix not found, skip this call
                call_idx = call_end + len(self.tool_call_end)
                continue
                
            # Extract JSON arguments
            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")
                
            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue
                
            args_str = call_content[json_start:json_end].strip()
            
            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                args_json = {"error": "Error decoding JSON arguments"}
                
            # Add this tool call to our list
            tool_calls.append({
                "name": function_name,
                "parameters": args_json
            })
            
            # Move past this call for the next iteration
            call_idx = call_end + len(self.tool_call_end)
        
        return tool_calls


def main():
    # Initialize the parser
    parser = R1ToolParser()
    
    # Example model output with tool calls
    model_output = """
    I'll help you search for information about Python programming.
    
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_search_web
    ```json
    {"query": "Python programming tutorial", "result_count": 5}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_search_web
    ```json
    {"query": "Python programming tutorial", "result_count": 3}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>
    """
    
    # Parse the tool calls
    tool_inputs = parser.parse_input(model_output)
    
    # Access the parsed tool calls
    print(f"Found {len(tool_inputs.inputs)} tool calls:")
    for i, tool_call in enumerate(tool_inputs.inputs, 1):
        print(f"\nTool call {i}:")
        print(f"Name: {tool_call.name}")
        print(f"Parameters: {tool_call.parameters}")
    
    # Example of creating multiple tool outputs
    tool_outputs = ToolOutputs(
        outputs=[
            ToolOutput(
                name="search_web",
                output={
                    "results": [
                        {"title": "Python Tutorial", "url": "https://www.python.org/about/gettingstarted/"},
                        {"title": "Learn Python", "url": "https://www.learnpython.org/"}
                    ]
                }
            ),
            ToolOutput(
                name="get_weather",
                output={
                    "location": "San Francisco",
                    "temperature": 18,
                    "unit": "celsius",
                    "condition": "Partly cloudy"
                }
            )
        ]
    )
    
    # Format the tool outputs
    formatted_output = parser.parse_output(tool_outputs)
    print("\nFormatted tool outputs:")
    print(formatted_output)

    lol = [
    {'role': 'system', 'content': 'You are R1. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': 'Thinking deeper...', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}}, 
        {'type': 'function', 'function': {'name': 'get_temperature_date', 'arguments': {'location': 'San Francisco, CA, USA', 'date': '2024-10-01'}}},
    ]},
    {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
    ]
    # use tokenizer to turn this into a string
    tokenizer = parser.tokenizer
    
    # Serialize the arguments dictionary to a JSON string before applying the chat template
    for message in lol:
        if message['role'] == 'assistant' and 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                if 'function' in tool_call and 'arguments' in tool_call['function']:
                    if isinstance(tool_call['function']['arguments'], dict):
                        tool_call['function']['arguments'] = json.dumps(tool_call['function']['arguments'])
    
    tokenized_lol = tokenizer.apply_chat_template(lol, tokenize=False, special_tokens=False)
    print(tokenized_lol)

if __name__ == "__main__":
    main()