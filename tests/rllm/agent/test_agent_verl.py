import ray
import numpy as np
import hydra
import os
from tqdm import tqdm

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd
import json

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from rllm.models.web_agent import WebAgent
from rllm.models.batch_agent import BatchAgent
from rllm.environments.browsergym.browsergym import BatchBrowserGym


def init_rollout_engine(config):
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    return wg


@hydra.main(config_path='config', config_name='verl_generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    config.model.path = "Qwen/Qwen2.5-0.5B-Instruct" #"meta-llama/Llama-3.1-8B-Instruct"

    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    rollout_engine = init_rollout_engine(config)

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

    agent = BatchAgent(rollout_engine=rollout_engine, engine_name="verl", tokenizer=tokenizer, agent_class=WebAgent, n_parallel_agents=2, env=env)
    
    trajectories = agent.interact_environment()
    env.close()


if __name__ == '__main__':
    main()
