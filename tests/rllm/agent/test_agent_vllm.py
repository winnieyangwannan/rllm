import os

import browsergym.core
import gymnasium as gym
import torch

from rllm.models.web_agent import WebAgent
from rllm.models.batch_agent import BatchAgent
from rllm.environments.browsergym.browsergym import BatchBrowserGym


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
    tasks = [
        {
            "start_url": "https://www.google.com/maps",
            "goal": "Locate a parking lot near the Brooklyn Bridge that open 24 hours. Review the user comments about it.",
        },
        {
            "start_url": "https://www.google.com/maps",
            "goal": "Locate a parking lot near the Brooklyn Bridge that open 24 hours. Review the user comments about it.",
        }
    ]
    env = BatchBrowserGym(
        tasks=tasks,
        batch_size=2,
    )
    output_dir = "./agent_batch_test"
    # Set output directory
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)


    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    engine, tokenizer, sampling_params = init_vllm_engine(model_path)

    agent = BatchAgent(rollout_engine=engine, engine_name="vllm", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=2, sampling_params=sampling_params, env=env)
    
    trajectories = agent.interact_environment()
        
    torch.save(trajectories, os.path.join(output_dir, 'evaluate_trajectories.pt'))
    env.close()

if __name__ == "__main__":
    main()
