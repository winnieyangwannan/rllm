import json
import asyncio
import time
from openai import AsyncOpenAI
from e2b_code_interpreter import AsyncSandbox

tools = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute python code in a sandbox and return result, good for simple python code like calculations and counting, and other basic math tasks",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute in a single cell"
                }
            },
            "required": ["code"]
        }
    }
}]


sandbox = None

async def _init_sandbox():
    global sandbox
    if sandbox is None:
        print("create sandbox") 
        sandbox = await AsyncSandbox.create(api_key="")  # need an API key here for e2b sandbox

async def _kill_sandbox():
    global sandbox
    if sandbox is not None:
        print("kill sandbox")
        await sandbox.kill()

async def _execute_python(code = ""):
    global sandbox
    print("Execute SANDBOX")
    execution = await sandbox.run_code(code)
    # format res to string
    print(execution)
    return str(execution)

NAME_TO_FUNCTION = {
    'execute_python' : _execute_python
}


def parse_tool_calls(tool_call_str):
    # Remove any whitespace and newlines
    tool_call_str = tool_call_str.strip()
    
    # First try to extract JSON between ```json ``` tags
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
        "id": "manual_tool_call" # Add default ID for manual tool calls
    }

# pass in completion from client.chat.completions.create()
# returns true if tool used and appends the tool response to messages
async def _apply_tool(completion, messages):
    if len(completion.choices[0].message.tool_calls) > 0:
        for tool_call in completion.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            function = NAME_TO_FUNCTION[tool_call.function.name]
            res = await function(**args)
            messages.append(completion.choices[0].message)
            messages.append({
                "role": "tool",
                "name": tool_call.function.name,
                "content": res,
                "tool_call_id": tool_call.id,
            })
        return True, messages
    elif "```json" in completion.choices[0].message.content:
        tool_call = parse_tool_calls(completion.choices[0].message.content)
        
        if tool_call:
            function = NAME_TO_FUNCTION[tool_call["name"]]
            res = await function(**tool_call["parameters"])
            messages.append({
                "role": "assistant",
                "content" : completion.choices[0].message.content,
            })
            # messages.append(completion.choices[0].message)
            messages.append({
                "role": "tool", 
                "name": tool_call["name"],
                "content": res,
                "tool_call_id": tool_call["id"]
            })
            return True, messages
    return False, messages

async def _cleanup():
    await _kill_sandbox()

async def _init():
    await _init_sandbox()


def chat_completion_with_tool(client: AsyncOpenAI, messages_list, model = "gpt-4o"):
    async def tool_call_flow(messages):
        completion = await client.chat.completions.create(
            model = model,
            messages = messages,
            tools = tools,
        )
        # print("first round:", completion.choices[0].message.content)

        use_tools, messages = await _apply_tool(completion, messages)
        print("messages", messages)
        if use_tools:
            completion_final = await client.chat.completions.create(
                model = model,
                messages = messages,
                tools = tools
            )
            return completion_final.choices[0].message.content
        
        return completion.choices[0].message.content
    
    async def run_batch():
        await _init()
        tasks = [tool_call_flow(messages) for messages in messages_list]
        result = await asyncio.gather(*tasks)
        await _cleanup()

        return result
    

    return asyncio.run(run_batch())

if __name__ == '__main__':    
    messages = [
        [{"role": "system", "content": ""}, {"role": "user", "content": "How what is the square root of 20?"}],
        [{"role": "system", "content": ""}, {"role": "user", "content": "Calculate 97^10."}]
    ]

    # This is for model served with vLLM.
    client = AsyncOpenAI(base_url="http://0.0.0.0:8080/v1")
    print(chat_completion_with_tool(client, messages, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"))
    
    
    # client = AsyncOpenAI(api_key='')
    # print(chat_completion_with_tool(client, messages, model="gpt-4o"))