import ray
import numpy as np
import hydra
import os
from tqdm import tqdm
import csv

import gymnasium as gym
import browsergym.webarena
import browsergym.miniwob 
from transformers import AutoTokenizer

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import pandas as pd
import json
from rllm.environments.browsergym import BatchBrowserGym

from rllm.models.web_agent import WebAgent
from rllm.rllm.models.agent_execution_engine import AgentExecutionEngine
from rllm.environments.browsergym.browsergym import BatchBrowserGym
import torch 

def main():
    number_of_tasks = 100
    seed = 42
    metric_file = "evaluate_metrics_openai.csv"
    trajectory_file = 'evaluate_trajectories_openai.pt'

    miniwob_url = "file://<PATH_TO_MINIWOB_CLONED_REPO>/miniwob/html/miniwob/"
    if "MINIWOB_URL" not in os.environ:
        os.environ["MINIWOB_URL"] = miniwob_url
        print(f"MINIWOB_URL set to {miniwob_url}")

    model_path = "Qwen/Qwen2.5-7B-Instruct-1M",
    # Init output dir
    output_dir = "miniwob_evaluator_openai_o1_preview"

    # Set output directory
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory set to: {output_dir}")

    # Init env
    env_ids = [id for id in gym.envs.registry.keys() if id.startswith("browsergym/miniwob")]

    rng = np.random.default_rng(seed)
    num_tasks = min(number_of_tasks, len(env_ids))
    selected_envs = rng.choice(env_ids, size=num_tasks, replace=False)
    env = BatchBrowserGym(
        env_id=selected_envs,
        batch_size=len(selected_envs),
    )
    api_key_path="keys.txt"
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_cache=False)
    # Init agent
    agent = AgentExecutionEngine(rollout_engine=None, engine_name="openai", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=len(selected_envs), api_key=api_key, env=env)
    timing_raw = {}
    evaluate_trajectories = agent.interact_environment(atiming_raw=timing_raw)

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
