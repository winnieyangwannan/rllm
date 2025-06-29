"""
Generates SFT data from DeepScaleR trajectories.
Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server   --model agentica-org/DeepScaleR-1.5B-Preview   --port 30000   --tensor-parallel-size 2
    python generate_sft_data.py --num_samples 500 --reward_threshold 0.5

"""

import argparse
import asyncio
import os

import pandas as pd

from rllm.trainer.agent_sft_trainer import AgentSFTTrainer


def load_problems(num_samples):
    """Load problems from dataset."""
    df = pd.read_parquet("../../data/train.parquet")
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)

    return [{"question": row["task"]["question"], "ground_truth": row["task"]["ground_truth"], "uid": row["uid"]} for _, row in df.iterrows()]


async def generate_trajectories(tasks):
    """Generate trajectories using DeepScaleR-1.5B."""
    from transformers import AutoTokenizer

    from rllm.agents.math_agent import MathAgent
    from rllm.engine import AsyncAgentExecutionEngine
    from rllm.environments.base.single_turn_env import SingleTurnEnvironment
    from rllm.rewards.reward_fn import math_reward_fn

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_name = "agentica-org/DeepScaleR-1.5B-Preview"

    engine = AsyncAgentExecutionEngine(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        engine_name="openai",
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        sampling_params={"temperature": 0.6, "top_p": 0.95, "model": model_name},
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None"},
        max_response_length=15000,
        max_prompt_length=2048,
        n_parallel_agents=256,
    )

    return await engine.execute_tasks(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--reward_threshold", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="sft_data.parquet")
    args = parser.parse_args()

    # Load and generate
    tasks = load_problems(args.num_samples)
    results = asyncio.run(generate_trajectories(tasks))

    # Process trajectories
    sft_data = AgentSFTTrainer.process_trajectories(results, args.reward_threshold)

    # Save results
    if sft_data:
        pd.DataFrame(sft_data).to_parquet(args.output, index=False)
        lengths = [len(" ".join([m["content"] for m in ex["messages"] if m["role"] == "assistant"])) for ex in sft_data]
        print(f"Saved {len(sft_data)} examples. Response lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) // len(lengths)}")
    else:
        print("No valid data generated!")


if __name__ == "__main__":
    main()
