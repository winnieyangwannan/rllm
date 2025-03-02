from datasets import load_dataset
import json
from tqdm import tqdm
import random

TOOLCALL_REWRITE_PROMPT = """You are a math assistant that shows your work step by step. 
    Given a thinking trajectory that has correct final answer, rewrite it to include Python tool calls to verify intermediate calculations.
    Use the python_interpreter function to verify calculations.
    Keep the LaTeX formatting and the original mathematical reasoning, but add tool calls where appropriate to verify results.
    
    Example tool call format:
    <tool_call>
    {{"name": "python"_interpreter, "parameters": {{"code": "from sympy import symbols, solve\\nx = symbols('x')\\nsolve(x**2 - 4, x)"}}}}
    </tool_call>

    Tool call response should be put in:
    <tool_responset>example tool call response</tool_response>
    
    Problem:
    {problem}

    Original trajectory:
    {trajectory}

    Please rewrite this trajectory to include Python tool calls for calculation verification. Follow these requirements:
    1. Keep the original mathematical reasoning and LaTeX formatting intact
    2. Add tool calls using <tool_call> tags to verify key calculations
    3. Wrap the complete thinking process in <think></think> tags
    4. After the thinking process:
       - answer the question by summarizing the key steps in your thinking.
       - Put the final answer in a \\boxed{{}} environment.
    5. Only call tools to verify calculations inside the <think></think> tags. After the thinking process, summarize and give the final answer without additional tool calls.
    6. Only call tools for complex calculations that are error-prone or difficult to verify by hand. For example:
       - Do not call tools for basic arithmetic like 100+200 or 5*3
       - Do call tools for calculations involving fractions, decimals, equations, or multiple steps
    7. Maintain the same level of detail as the original trajectory.
    
    The tool calls should follow this exact format:
    <tool_call>
    {{"name": "python_interpreter", "parameters": {{"code": "your_python_code_here"}}}}
    </tool_call>
    
    Tool responses should be wrapped in <tool_response> tags."""


def rewrite_trajectory(problem, generation, client, model):
    response = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": TOOLCALL_REWRITE_PROMPT.format(problem=problem, trajectory=generation)}
        ],
        max_tokens=8192,
    )
    

    return response.content[0].text
    # return response.choices[0].message.content


if __name__ == "__main__":
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

    print("Available sources:", set(ds['train']['source']))
    ds = ds['train'].filter(lambda x: x['source'] in ['olympiads', 'amc_aime'])

    filtered_problems = []

    for item in ds:
        generations = item['generations']
        for idx, gen in enumerate(generations):
            # Check if reasoning is complete and correct
            if item['is_reasoning_complete'][idx] and item['correctness_math_verify'][idx]:
                problem_dict = {
                    'problem': item['problem'],
                    'answer': item['answer'],
                    'reasoning': gen,
                }
                filtered_problems.append(problem_dict)
                break  # Take only the first correct and complete reasoning for each problem

    print(f"Total filtered problems: {len(filtered_problems)}")

    ds = [p for p in filtered_problems if len(p['reasoning']) <= 5000]
    print(f"Problems after length filtering: {len(ds)}")
    ds = [p for p in ds if len(p['reasoning']) > 3000]
    print(f"Problems after length filtering: {len(ds)}")

    random.seed(42)
    random.shuffle(ds)

    from openai import OpenAI
    client = OpenAI(api_key="")

    import anthropic
    client = anthropic.Anthropic(
        api_key="",
    )
    model = "claude-3-7-sonnet-20250219"

    for example in tqdm(ds):
        enhanced = rewrite_trajectory(example['problem'], example['reasoning'], client, model)
        example['reasoning_with_toolcall'] = enhanced
        with open('./data/tool_calls_claude.jsonl', 'a') as f:
            f.write(json.dumps(example) + '\n')