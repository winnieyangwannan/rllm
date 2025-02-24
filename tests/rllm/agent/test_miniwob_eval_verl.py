import ray
import numpy as np
import hydra
import os
from tqdm import tqdm

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import gymnasium as gym
import csv
import torch

from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from rllm.models.web_agent import WebAgent
from rllm.models.batch_agent import BatchAgent
from rllm.environments.browsergym.browsergym import BatchBrowserGym
import os
import browsergym.miniwob

def init_rollout_engine(config):
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    return wg


@hydra.main(config_path='config', config_name='verl_generation', version_base=None)
def main(config):
    number_of_tasks = 2
    seed = 42
    safe_batch_size = 64
    episode_len = 2
    metric_file = "evaluate_metrics_verl.csv"
    trajectory_file = 'evaluate_trajectories_verl.pt'

    miniwob_url = "file://<PATH_TO_MINIWOB_CLONED_REPO>/miniwob/html/miniwob/"
    if "MINIWOB_URL" not in os.environ:
        os.environ["MINIWOB_URL"] = miniwob_url
        print(f"MINIWOB_URL set to {miniwob_url}")

    model_path = "Qwen/Qwen2.5-0.5B-Instruct" #"Qwen/Qwen2.5-7B-Instruct-1M"
    # Init output dir
    output_dir = "miniwob_evaluator_verl_test"
    
    # Init verl config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    config.model.path = model_path

    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rollout_engine = init_rollout_engine(config)

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

    # Init agent
    agent = BatchAgent(rollout_engine=rollout_engine, engine_name="verl", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=len(selected_envs), safe_batch_size=safe_batch_size, episode_len=episode_len, env=env)

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
