import asyncio

import numpy as np
from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv


def load_frozenlake_data(n=1, train_size=3000, test_size=100):
    # Check if dataset already exists in registry
    if DatasetRegistry.dataset_exists("frozenlake", "test"):
        test_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
        return test_dataset.get_data()
    
    # If not, create and register the dataset
    np.random.seed(42)
    train_seeds = np.random.randint(0, 100000, size=train_size)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    train_sizes = np.random.randint(2, 10, size=train_size)
    test_sizes = np.random.randint(2, 10, size=test_size)
    train_ps = np.random.uniform(0.6, 0.85, size=train_size)
    test_ps = np.random.uniform(0.6, 0.85, size=test_size)

    def frozenlake_process_fn(seed, size, p, idx):
        return {
            "seed": seed,
            "size": size,
            "p": p,
            "index": idx,
            "uid": f"{seed}_{size}_{p}"
        }

    train_data = [frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx) for idx, seed in enumerate(train_seeds)]
    test_data = [frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx) for idx, seed in enumerate(test_seeds)]

    # Register the datasets with separate splits
    DatasetRegistry.register_dataset("frozenlake", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("frozenlake", test_data, "test")
    
    return test_dataset.get_data()

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    agents = [FrozenLakeAgent() for _ in range(n_parallel_agents)]
    envs = [FrozenLakeEnv() for _ in range(n_parallel_agents)]

    engine = AsyncAgentExecutionEngine(
        agents=agents,
        envs=envs,
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        config=None,
        n_parallel_agents=n_parallel_agents,
        enable_thinking=True,
    )

    tasks = load_frozenlake_data(n=1)

    results = asyncio.run(engine.execute_tasks(tasks))
