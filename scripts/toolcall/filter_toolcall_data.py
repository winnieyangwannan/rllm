from datasets import load_dataset
import re
import json
from pprint import pprint
from tqdm import tqdm
from rllm.environments.tools import PythonInterpreter
from openai import AsyncOpenAI

def parse_trajectory(example):
    """Parse a trajectory with tool calls into OpenAI chat format."""
    # Initialize messages list with system message
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": example['problem']}]
    
    text = example['reasoning_with_toolcall']
    # Extract content after <think> tag
    think_start = text.find('<think>')
    if think_start != -1:
        text = text[think_start:]
    
    # Split content by tool calls
    # Look for JSON blocks between <tool_call> and </tool_call> markers
    parts = re.split(r'<tool_call>(.*?)</tool_call>', text, flags=re.DOTALL)
    
    current_content = ""
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Regular content
            # Remove tool call output sections
            part = re.sub(r'<tool_response>.*?</tool_response>', '', part, flags=re.DOTALL)
            current_content += part
        else:  # Tool call
            # Add accumulated assistant message if any
            if current_content.strip():
                messages.append({
                    "role": "assistant",
                    "content": current_content.strip()
                })
                current_content = ""
            
            # Parse the tool call JSON
            try:
                tool_call = json.loads(part)
                messages.append({
                    "role": "assistant", 
                    "content": None,
                    "function_call": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["parameters"])
                    }
                })
                
                # Find tool output in next part
                next_part = parts[i+1] if i+1 < len(parts) else ""
                tool_output_match = re.search(r'<tool_response>(.*?)</tool_response>', 
                                           next_part, flags=re.DOTALL)
                tool_output = tool_output_match.group(1) if tool_output_match else "<no tool output found>"
                
                # Add the actual tool response
                messages.append({
                    "role": "function",
                    "name": tool_call["name"], 
                    "content": tool_output.strip()
                })
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse tool call JSON: {part}")
                return []
    
    # Add any remaining content as final assistant message
    if current_content.strip():
        messages.append({
            "role": "assistant",
            "content": current_content.strip()
        })
    
    return messages


def get_rewrite_prompt(code, error_msg):
    """
    Generates a prompt for the LLM to fix incorrect code based on error message.
    """
    prompt = f"""You are a Python programming expert. Fix the following code that produced an error.
    The code should perform the same mathematical calculation but with correct syntax.
    
    Original code:
    ```python
    {code}
    ```
    
    Error message:
    {error_msg}
    
    Please provide only the corrected code with no additional explanation.
    Keep imports if they were present in the original code.
    Ensure the code performs the same mathematical operation as intended."""
    
    return prompt


async def rewrite_code_with_llm(code, error_msg, client):
    """
    Uses GPT to rewrite code that produced errors.
    Returns the rewritten code.
    """
    prompt = get_rewrite_prompt(code, error_msg)
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    code = response.choices[0].message.content.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


async def verify_tool_calls(messages, interpreter, client):
    has_error = False
    
    for msg in messages:
        if msg.get("function_call"):
            try:
                args = json.loads(msg["function_call"]["arguments"])                
                if "code" in args:
                    # print("\nExecuting code:", args["code"])
                    result = await interpreter.execute(code=args["code"])
                    
                    # Check if result contains error
                    if isinstance(result, str) and 'Error' in result:
                        has_error = True
                        print(result)

                        new_code = await rewrite_code_with_llm(args["code"], result, client=client)
                        print("new_code", new_code)
                        
                        result = await interpreter.execute(code=new_code)
                        print("result after new code:", result)
                        break
                    
                    # Find and update the corresponding function response
                    idx = messages.index(msg)
                    if idx + 1 < len(messages) and messages[idx + 1]["role"] == "function":
                        original_content = messages[idx + 1]["content"]
                        messages[idx + 1].update({
                            "content": str(result),
                            "original_content": original_content
                        })
            except Exception as e:
                has_error = True
                print(f"Error executing code: {e}")
                break
    
    return None if has_error else messages


if __name__ == "__main__":
    import os
    import asyncio

    trajectories = []
    count =0 
    with open('./data/tool_calls_raw.jsonl', 'r') as f:
        for line in tqdm(f):
            example = json.loads(line)
            messages = parse_trajectory(example)  # Using the existing parse_trajectory function
            if len(messages) == 0:
                continue
            trajectories.append({
                'problem': example['problem'],
                'answer': example['answer'],
                'messages': messages
            })

    print(len(trajectories))

    os.environ['E2B_API_KEY'] = ""

    interpreter = PythonInterpreter()

    client = AsyncOpenAI(api_key="")

    # Process all trajectories
    async def process_trajectories():
        filtered_trajectories = []
        for example in tqdm(trajectories[9:]):
            messages = await verify_tool_calls(example['messages'], interpreter)
            if messages is not None:
                filtered_trajectories.append({
                    'problem': example['problem'],
                    'answer': example['answer'],
                    'messages': messages
                })
        print(f"Filtered trajectories: {len(filtered_trajectories)}")

        with open('./data/filtered_toolcall.jsonl', 'w') as f:
            for trajectory in filtered_trajectories:
                f.write(json.dumps(trajectory) + '\n')

    asyncio.run(process_trajectories())
