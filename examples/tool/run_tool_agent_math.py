import asyncio

from rllm.agents import ToolAgent
from rllm.environments.tools.tool_env import ToolEnvironment

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import load_dataset
from copy import deepcopy
from transformers import AutoTokenizer

from rllm.data.utils import fetch_live_code_bench_system_prompt

from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    if DatasetRegistry.dataset_exists(
        "deepscaler_math"
    ) and DatasetRegistry.dataset_exists("aime2024"):
        train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
        test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
        return train_dataset, test_dataset

    train_dataset = load_dataset(
        "agentica-org/DeepScaleR-Preview-Dataset", split="train"
    )
    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset(
        "deepscaler_math", train_dataset, "train"
    )
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    return train_dataset, test_dataset


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
    n_parallel_agents = 64

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
        max_response_length=8192,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    _, test_dataset = prepare_math_data()
    tasks = test_dataset.repeat(n=1)

    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_results(results)