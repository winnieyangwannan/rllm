"""
run `pip install "gymnasium[toy-text]"` to install gymnasium
"""

"""
Adapted from nice codes from gymnasium.envs.toy_text.frozen_lake.generate_random_map
Modify it so that the start and end points are random
"""

import gymnasium as gym
from typing import List, Optional, Tuple, Any, Dict
import hashlib
import numpy as np
import copy
from ..batch_env import BatchedEnv
from r2e_edits.agenthub.environment.env import EnvArgs, RepoEnv
from r2e_edits.agenthub.action import Action
from r2e_edits.agenthub.runtime.docker import DockerRuntime

class SWEEnv:
    def __init__(self, **kwargs):

        # Get all available images
        from datasets import load_dataset
        self.available_dataset = load_dataset("r2e-edits/swebench-verified-v1", split="test")
        with open("./command_files.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.command_files = config["command_files"]
        self.max_steps = max_steps
    
    def reset(self):
        select_idx = random.choice(range(len(self.available_dataset)))
        env_args = EnvArgs(ds = self.available_dataset[select_idx])
        self.env = RepoEnv(eng_args)

        # reset environment
        self.env.reset()
        self.env.runtime.reset()
        self.env.runtime.setup_env()
        self.env.add_commands(self.command_files)
        self.env.runtime.start_new_branch()
        self.total_steps = 0

        return self.env.runtime.get_task_instruction()

    def evaluate_reward(self):
        reward, test_output = self.env.runtime._calculate_reward(get_test_output = True)
        return reward
        
    def step(self, action: Action):
        # Check for max steps
        if self.total_steps > self.max_steps:
            return "Max Time steps", 0, True, True, {}

        if action.function_name == "":
            return "", 0, False, False, {}

        obs, reward, done info = self.env.step(action, timeout = 20)

        self.total_steps += 1
        return obs, reward, done, False, {}

    def close(self):
        self.env.close()
     
class BatchSWEEnv(BatchedEnv):
    def __init__(
        self,
        batch_size,
        seeds, 
        sizes,
        ps,
    ):
        self.envs = []
        self._env_id = []
        for i in range(batch_size):
            seed = seeds[i]
            size = sizes[i]
            p = ps[i]
            self.envs.append(FrozenLakeEnv(size=size, seed=seed, p=p))
            self._env_id.append(f"{seed}-{size}-{p}")

        self._batch_size = batch_size

    @property
    def env_id(self) -> List[str]:
        return self._env_id

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset(self, seed=0) -> Tuple[List, List]:
        observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset(reset_map=False)
            observations.append(obs)
        return observations, [{}] * self.batch_size

    def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:
        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"

        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        # Send step command with actions
        for i, env_idx in enumerate(env_idxs):
            obs, reward, done, info = self.envs[env_idx].step(actions[i])
            observations.append(obs),
            rewards.append(reward)
            terminateds.append(done)
            truncateds.append(False)
            infos.append(info)

        return (observations, rewards, terminateds, 
                truncateds, infos)

    def close(self):
        for i, env in enumerate(self.envs):
            env.close()

