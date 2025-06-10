import asyncio

from transformers import AutoTokenizer

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn

def evaluate_results(results):
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        problem = trajectory.steps[0].observation

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem] += is_correct
        problem_total_map[problem] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = (
        sum(1 for problem, correct in problem_correct_map.items() if correct > 0)
        / total_problems
    )

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = math_reward_fn

    envs = [SingleTurnEnvironment(reward_fn=reward_fn) for _ in range(n_parallel_agents)]

    agents = [MathAgent() for i in range(n_parallel_agents)]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

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
        max_response_length=32768,
        max_prompt_length=2048,
        config=None,
        n_parallel_agents=n_parallel_agents,
        enable_thinking=True,
    )

    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from .prepare_math_data import prepare_math_data
        _, test_dataset = prepare_math_data()
    
    tasks = test_dataset.repeat(n=16)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_results(results)
