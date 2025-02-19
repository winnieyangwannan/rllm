import json

def parse_tool_calls(tool_call_str):
    tool_call_str = tool_call_str.strip()

    # Try to extract JSON between ```json ``` tags
    if "```json" in tool_call_str:
        try:
            json_str = tool_call_str.split("```json")[1].split("```")[0].strip()
            tool_call = json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            # If extraction fails, try parsing the whole string
            try:
                tool_call = json.loads(tool_call_str)
            except json.JSONDecodeError:
                return None

    # Extract name and parameters
    name = tool_call.get("name")
    parameters = tool_call.get("parameters", {})

    return {
        "name": name,
        "parameters": parameters,
        "id": "manual_tool_call",  # Add default ID for manual tool calls
        "json_str": json_str
    }