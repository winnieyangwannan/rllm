import asyncio
import argparse

import numpy as np
from transformers import AutoTokenizer

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.utils import compute_pass_at_k
from prepare_miniwob_data import prepare_miniwob_data


def load_miniwob_data():
    if DatasetRegistry.dataset_exists("miniwob", "test"):
        test_dataset = DatasetRegistry.load_dataset("miniwob", "test")
        return test_dataset.get_data()
    
    print("MiniWoB datasets not found. Preparing datasets...")
    train_dataset, test_dataset = prepare_miniwob_data()
    
    return test_dataset.get_data()

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--stepwise", action="store_true", help="Run in stepwise mode")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    if not args.stepwise:
        engine = AsyncAgentExecutionEngine(
            agent_class=MiniWobAgent,
            env_class=BrowserGymEnv,
            agent_args={
                "use_accumulate_thinking": True,
                "use_full_conversation": True,
            },
            env_args={},
            engine_name="openai",
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            rollout_engine_args={
                "base_url": "http://localhost:30000/v1",
                "api_key": "None",
            },
            max_response_length=16384,
            max_prompt_length=3072,
            n_parallel_agents=n_parallel_agents,
            disable_thinking=False,
            enforce_max_prompt_length=False,
        )
    else:
        engine = AsyncAgentExecutionEngine(
            agent_class=MiniWobAgent,
            env_class=BrowserGymEnv,
            agent_args={
                "use_accumulate_thinking": False,
                "use_full_conversation": False,
            },
            env_args={},
            engine_name="openai",
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            rollout_engine_args={
                "base_url": "http://localhost:30000/v1",
                "api_key": "None",
            },
            max_response_length=3072,
            max_prompt_length=16384,
            n_parallel_agents=n_parallel_agents,
            disable_thinking=False,
            enforce_max_prompt_length=True
        )

    tasks = load_miniwob_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)