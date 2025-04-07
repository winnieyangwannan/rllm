import os
import csv
import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import json

from rllm.environments.browsergym import BatchBrowserGym
from rllm.models.web_agent import WebAgent
from rllm.rllm.models.agent_execution_engine import AgentExecutionEngine

import importlib
import browsergym.miniwob

importlib.reload(browsergym.miniwob)

def init_vllm_engine(model_name):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_cache=False)
    engine = LLM(
        model=model_name,
        tensor_parallel_size=8,
        enforce_eager=False,
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
    number_of_tasks = 100
    episode_len = 20
    dataset_file_path = "/home/colin/data/rllm-miniwob/train.parquet"
    output_file_path = "./sft_trajectory.json"

    teacher_model_path = "Qwen/Qwen2.5-72B-Instruct"

    dataset = pd.read_parquet(dataset_file_path)
    env = BatchBrowserGym.from_extra_infos(dataset["extra_info"].tolist()[:number_of_tasks])

    engine, tokenizer, sampling_params = init_vllm_engine(teacher_model_path)
    agent = AgentExecutionEngine(rollout_engine=engine, engine_name="vllm", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=env.batch_size, episode_len=episode_len, sampling_params=sampling_params, env=env, model_path=teacher_model_path)
    
    timing_raw = {}
    evaluate_trajectories = agent.interact_environment(timing_raw=timing_raw, mode="Conversation")

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(evaluate_trajectories, f, indent=4, ensure_ascii=False)
    print("Trajectory saved")
    env.close()


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()

