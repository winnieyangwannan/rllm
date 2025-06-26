"""
Generates SFT data from DeepScaleR trajectories.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server   --model agentica-org/DeepScaleR-1.5B-Preview   --port 30001   --tensor-parallel-size 2
    python generate_sft_data.py --num_samples 500
"""

import argparse
import asyncio
import os
import re

import pandas as pd


def load_problems(num_samples):
    """Load problems from DeepScaleR dataset."""
    print(f"Loading {num_samples} problems...")
    df = pd.read_parquet("../../data/train.parquet")
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)

    tasks = []
    for _, row in df.iterrows():
        tasks.append({"question": row["task"]["question"], "ground_truth": row["task"]["ground_truth"], "uid": row["uid"]})

    print(f"Loaded {len(tasks)} problems")
    return tasks


async def generate_trajectories(tasks):
    """Generate trajectories using DeepScaleR-1.5B."""
    print("Generating trajectories...")

    from transformers import AutoTokenizer

    from rllm.agents.math_agent import MathAgent
    from rllm.engine import AsyncAgentExecutionEngine
    from rllm.environments.base.single_turn_env import SingleTurnEnvironment
    from rllm.rewards.reward_fn import math_reward_fn

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine = AsyncAgentExecutionEngine(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={"temperature": 0.35, "top_p": 0.95, "model": model_name},
        rollout_engine_args={"base_url": "http://localhost:30001/v1", "api_key": "None"},
        max_response_length=15000,
        max_prompt_length=2048,
        n_parallel_agents=256,
    )

    results = await engine.execute_tasks(tasks)
    print(f"Generated {len(results)} trajectories")
    return results


def extract_messages_from_result(result):
    """Extract messages from result object."""
    # Try direct chat_completions first
    if hasattr(result, "chat_completions") and result.chat_completions:
        return result.chat_completions

    # Try trajectory[-1].chat_completions
    if hasattr(result, "trajectory") and result.trajectory:
        if isinstance(result.trajectory, list) and result.trajectory:
            return result.trajectory[-1].chat_completions
        else:
            return result.trajectory.chat_completions

    # Try steps[0].chat_completions (for Trajectory objects)
    if hasattr(result, "steps") and result.steps and len(result.steps) > 0:
        return result.steps[0].chat_completions

    return None


def process_trajectories(results, tasks, correct_only=False):
    """Process trajectories into SFT format."""
    print("Processing trajectories...")

    sft_data = []
    valid = 0
    correct = 0

    for result, task in zip(results, tasks, strict=False):
        if result is None:
            continue

        messages = extract_messages_from_result(result)
        if not messages:
            continue

        # Ensure messages have the right format
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = str(msg["content"]).strip()
                if content:
                    clean_messages.append({"role": msg["role"], "content": content})

        if len(clean_messages) < 2:
            continue

        valid += 1

        # Check if answer is correct (optional filtering)
        if correct_only:
            assistant_responses = [m["content"] for m in clean_messages if m["role"] == "assistant"]
            if assistant_responses:
                # Extract boxed answer
                matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", assistant_responses[-1])
                predicted = matches[-1] if matches else ""

                if predicted.strip() == task["ground_truth"].strip():
                    correct += 1
                else:
                    continue  # Skip incorrect answers
            else:
                continue

        sft_data.append({"messages": clean_messages})

    print(f"Results: {valid} valid trajectories, {len(sft_data)} final examples")
    if correct_only:
        print(f"Correct answers: {correct}")

    return sft_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="Number of problems to generate trajectories for")
    parser.add_argument("--correct_only", action="store_true", help="Only keep trajectories with correct answers")
    parser.add_argument("--output", type=str, default="sft_data.parquet", help="Output file path")
    args = parser.parse_args()

    print(f"Generating SFT data: {args.num_samples} samples -> {args.output}")

    # Load problems
    tasks = load_problems(args.num_samples)

    # Generate trajectories
    results = asyncio.run(generate_trajectories(tasks))

    # Process and save
    sft_data = process_trajectories(results, tasks, args.correct_only)

    if sft_data:
        # Save to parquet format for MultiTurnSFTDataset
        df = pd.DataFrame(sft_data)
        df.to_parquet(args.output, index=False)
        print(f"✅ Saved {len(sft_data)} examples to {args.output}")

        # Statistics
        lengths = [len(" ".join([m["content"] for m in example["messages"] if m["role"] == "assistant"])) for example in sft_data]
        print(f"Assistant response lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) // len(lengths)}")
    else:
        print("❌ No valid SFT data generated!")


if __name__ == "__main__":
    main()
