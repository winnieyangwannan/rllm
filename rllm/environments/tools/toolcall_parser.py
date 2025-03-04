import json

class ToolcallParser:
    """Parser for different tool call formats."""
    
    PARSERS = {
        "r1": "parse_r1_tool_calls",
        "qwen": "parse_qwen_tool_calls",
        "json": "parse_json_tool_calls",
        "python": "parse_python_tool_calls"
    }
    
    def __init__(self, parser_type: str = "json"):
        """Initialize the parser with specified type.
        
        Args:
            parser_type (str): Type of parser to use ('r1', 'qwen', or 'default')
        """
        if parser_type not in self.PARSERS:
            raise ValueError(f"Unknown parser type: {parser_type}. Available types: {list(self.PARSERS.keys())}")
        self.parser_type = parser_type

    def parse(self, text: str) -> list[dict]:
        """Parse tool calls using the configured parser.
        
        Args:
            text (str): Text containing tool calls
            
        Returns:
            list[dict]: List of parsed tool calls
        """
        parser_method = getattr(self, self.PARSERS[self.parser_type])
        return parser_method(text)

    @staticmethod
    def parse_r1_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from text using the special token format.

        Format:
        <｜tool▁calls▁begin｜>
            <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
            ```json
            {"param": "value"}
            ```
        <｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
        """
        TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
        TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
        TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
        TOOL_CALL_END = "<｜tool▁call▁end｜>"
        TOOL_SEP = "<｜tool▁sep｜>"
        
        tool_calls = []
        
        # Return empty list if no tool calls section found
        if TOOL_CALLS_BEGIN not in text:
            return tool_calls
            
        # Extract the tool calls section
        start = text.find(TOOL_CALLS_BEGIN) + len(TOOL_CALLS_BEGIN)
        end = text.find(TOOL_CALLS_END)
        if end == -1:
            return tool_calls
        
        tool_calls_text = text[start:end].strip()
        
        # Split into individual tool calls
        while TOOL_CALL_BEGIN in tool_calls_text:
            # Extract one tool call
            call_start = tool_calls_text.find(TOOL_CALL_BEGIN) + len(TOOL_CALL_BEGIN)
            call_end = tool_calls_text.find(TOOL_CALL_END)
            if call_end == -1:
                break
                
            call_text = tool_calls_text[call_start:call_end].strip()
            
            # Parse function name
            func_sep_idx = call_text.find(TOOL_SEP)
            if func_sep_idx == -1:
                break
            function_name = call_text[func_sep_idx + len(TOOL_SEP):].split('\n')[0].strip()
            
            # Extract JSON arguments
            json_start = call_text.find("```json\n") + len("```json\n")
            json_end = call_text.find("```", json_start)
            if json_start == -1 or json_end == -1:
                break
                
            arguments_str = call_text[json_start:json_end].strip()

            try:
                arguments_json = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments_json = {"error": "Error decoding the json arguments"}

            tool_calls.append({
                "name": function_name,
                "parameters": arguments_json
            })
            # Move to next tool call
            tool_calls_text = tool_calls_text[call_end + len(TOOL_CALL_END):]
        
        return tool_calls


    @staticmethod
    def parse_qwen_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from text using a simple token format.
    
        Format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
        
        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        TOOL_CALL_BEGIN = "<tool_call>"
        TOOL_CALL_END = "</tool_call>"
        
        tool_calls = []
        
        # Return empty list if no tool calls found
        if TOOL_CALL_BEGIN not in text:
            return tool_calls
            
        # Process all tool calls in the text
        while TOOL_CALL_BEGIN in text:
            start = text.find(TOOL_CALL_BEGIN) + len(TOOL_CALL_BEGIN)
            end = text.find(TOOL_CALL_END)
            if end == -1:
                break
                
            # Extract and parse the JSON content
            json_content = text[start:end].strip()
            try:
                call_data = json.loads(json_content)
                # Convert to common format matching parse_tool_calls output
                tool_calls.append({
                    "name": call_data["name"],
                    "parameters": call_data["arguments"]
                })
            except json.JSONDecodeError:
                break
                
            # Move to next potential tool call
            text = text[end + len(TOOL_CALL_END):]
        
        return tool_calls

    @staticmethod
    def parse_json_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from text using JSON code block format.
        
        Format:
        ```json
        {"name": "function_name", "arguments": {...}}
        ```
        
        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []
        
        # Find all occurrences of ```json blocks
        while "```json" in text:
            # Find start of JSON block
            start = text.find("```json") + len("```json")
            # Find end of JSON block
            end = text.find("```", start)
            if end == -1:
                break
                
            # Extract and parse the JSON content
            json_content = text[start:end].strip()
            try:
                call_data = json.loads(json_content)
                # Convert to common format matching parse_tool_calls output
                tool_calls.append({
                    "name": call_data["name"],
                    "parameters": call_data["arguments"]
                })
            except Exception as e:
                try:
                    # Replace single quotes with double quotes
                    json_content = json_content.replace("'", '"')
                    # Fix unquoted keys
                    json_content = json_content.replace('{', '{"').replace(':', '":').replace(', ', ',"').replace('"}', '}')
                    # Remove any trailing commas before closing brackets
                    json_content = json_content.replace(',}', '}').replace(',]', ']')
                    
                    call_data = json.loads(json_content)
                    tool_calls.append({
                        "name": call_data["name"],
                        "parameters": call_data["arguments"]
                    })
                except:
                    tool_calls.append({
                        "name": "error",
                        "parameters": {"error_log": str(e)}
                    })
                    print(f"Failed to parse JSON: {e}")
                
            # Move to next potential JSON block
            text = text[end + 3:]
        
        return tool_calls

    @staticmethod
    def parse_python_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from text using Python code block format.
        
        Format:
        ```python
        some_python_code_here
        ```
        
        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []
        
        # Find all occurrences of ```python blocks
        while "```python" in text:
            # Find start of Python block
            start = text.find("```python") + len("```python")
            # Find end of Python block
            end = text.find("```", start)
            # if end == -1:
            #     break
                
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