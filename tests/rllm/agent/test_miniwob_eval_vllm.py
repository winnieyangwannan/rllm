import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import csv
import json
import numpy as np
import torch
import gymnasium as gym
import browsergym.miniwob 

from rllm.environments.browsergym import BatchBrowserGym
from rllm.models.web_agent import WebAgent
from rllm.models.batch_agent import BatchAgent

def init_vllm_engine(model_name):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_cache=False)
    engine = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )

    return engine, tokenizer, sampling_params

def main():
    number_of_tasks = 2
    seed = 42
    safe_batch_size = 64
    episode_len = 2
    metric_file = "evaluate_metrics_vllm.csv"
    trajectory_file = 'evaluate_trajectories_vllm.pt'

    miniwob_url = "file://<PATH_TO_MINIWOB_CLONED_REPO>/miniwob/html/miniwob/"
    if "MINIWOB_URL" not in os.environ:
        os.environ["MINIWOB_URL"] = miniwob_url
        print(f"MINIWOB_URL set to {miniwob_url}")

    model_path = "Qwen/Qwen2.5-7B-Instruct-1M"
    # Init output dir
    output_dir = "miniwob_evaluator_vllm_test"
    
    # Set output directory
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory set to: {output_dir}")

    env_ids = [id for id in gym.envs.registry.keys() if id.startswith("browsergym/miniwob")]

    rng = np.random.default_rng(seed)
    num_tasks = min(number_of_tasks, len(env_ids))
    selected_envs = rng.choice(env_ids, size=num_tasks, replace=False)

    env = BatchBrowserGym(
        env_id=selected_envs,
        batch_size=len(selected_envs),
    )

    engine, tokenizer, sampling_params = init_vllm_engine(model_path)
    agent = BatchAgent(rollout_engine=engine, engine_name="vllm", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=len(selected_envs), safe_batch_size=safe_batch_size, episode_len=episode_len, sampling_params=sampling_params, env=env)
    
    timing_raw = {}
    evaluate_trajectories = agent.interact_environment(timing_raw=timing_raw)

    evaluate_metrics = {
        "evaluate_rollout.mean": np.mean([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
        "evaluate_rollout.max": np.max([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
        "evaluate_rollout.min": np.min([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
        "total_get_actions": timing_raw["get_actions_accum"],
        "total_env_step": timing_raw["env_step_accum"],
    }

    print(evaluate_metrics)

    # Save to CSV file
    with open(os.path.join(output_dir, metric_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in evaluate_metrics.items():
            writer.writerow([key, value])

    print("Metrics saved")
    torch.save(evaluate_trajectories, os.path.join(output_dir, trajectory_file))
    print("Trajectory saved")
    env.close()


if __name__ == "__main__":
    main()
