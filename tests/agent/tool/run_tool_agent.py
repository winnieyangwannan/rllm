import asyncio

from rllm.agents import ToolAgent
from rllm.environments.tools.tool_env import ToolEnvironment

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import load_dataset
from copy import deepcopy
from transformers import AutoTokenizer

from rllm.data.utils import fetch_live_code_bench_system_prompt


def load_data(n=1, dataset_enum=None):
    dataset = load_dataset(dataset_enum)
    data = []
    for idx, example in enumerate(dataset):
        if isinstance(dataset_enum, TestDataset.Math):
            processed = process_math_fn(example, idx)
        else:
            processed = process_code_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_math_fn(example, idx):
    print(example)
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}, and write python to evaluate math expressions if needed."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    task = {
        "ground_truth": answer,
        "question": question,
        "idx": idx,
        'data_source': 'math' 
    }
    return task


def process_code_fn(example, idx):
    # print(example)
    question = example.pop("problem")
    instruction = fetch_live_code_bench_system_prompt(prompt=question, starter_code=example.pop("starter_code"))

    question = f"{instruction}. You have access to a python interpreter. You can use it to write code and test it before outputting your final answer."
    ground_truth = example.pop("tests")

    task = {
        "ground_truth": ground_truth,
        "question": instruction,
        "idx": idx,
        'data_source': 'livecodebench' 
    }
    return task


def evaluate_results(results):
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        problem = trajectory.steps[0].observation['question']
        
        # Get is_correct directly from the trajectory's reward
        is_correct = 1 if trajectory.reward > 0 else 0
        
        problem_correct_map[problem] += is_correct
        problem_total_map[problem] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_16 = (
        sum(1 for problem, correct in problem_correct_map.items() if correct > 0)
        / total_problems
    )

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@16 Accuracy:", pass_at_16)
    

if __name__ == "__main__":
    # Create the environment (no batch_size parameter)
    n_parallel_agents = 1

    model_name = "Qwen/Qwen3-4B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    envs = [
        ToolEnvironment(tools=["python"]) for _ in range(n_parallel_agents)
    ]

    agents = [
        ToolAgent(tools=envs[i].tools.tools, model_name=model_name, parser_name='qwen') for i in range(n_parallel_agents)
    ]

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "tools": envs[0].tools.json, 
        "model": model_name
    }
    
    from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

    engine = AsyncAgentExecutionEngine(
        agents=agents,
        envs=envs,
        rollout_engine=None,
        engine_name="openai", 
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        max_response_length=32768,
        max_prompt_length=2048,
    )
    # engine.update_envs_and_agents(envs, agents)

    tasks = load_data(n=1, dataset_enum=TestDataset.Code.LIVECODEBENCH)

    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_results(results)