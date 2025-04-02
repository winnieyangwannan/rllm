import gymnasium as gym
from typing import List, Optional, Tuple, Any, Dict
import hashlib
import numpy as np
import copy
from ..batch_env import BatchedEnv
from r2e_edits.agenthub.environment.env import EnvArgs, RepoEnv
from r2e_edits.agenthub.action import Action
from r2e_edits.agenthub.runtime.docker import DockerRuntime
import random
from datasets import load_dataset
import os
import concurrent.futures

class SWEEnv:
    def __init__(self, dataset):

        # Get all available images
        # self.available_dataset = load_dataset("r2e-edits/swebench-verified-v1", split="test")
        self.available_dataset = dataset
        # TODO: Limit the number to 100 now
        self.available_dataset = self.available_dataset.select(range(100))

        command_files = [
                "../r2e-edits-internal/src/r2e_edits/agenthub/tools/file_editor.py",
                "../r2e-edits-internal/src/r2e_edits/agenthub/tools/search.py",
                "../r2e-edits-internal/src/r2e_edits/agenthub/tools/execute_bash.py",
                "../r2e-edits-internal/src/r2e_edits/agenthub/tools/finish.py",
        ]
        self.command_files = command_files
        self.max_steps = 20
        self.env = None
    
    def reset(self):
        select_idx = random.choice(range(len(self.available_dataset)))
        env_args = EnvArgs(ds = self.available_dataset[select_idx])
        self.env = RepoEnv(env_args)

        # reset environment
        self.env.reset()
        self.env.add_commands(self.command_files)
        self.total_steps = 0

        return self.env.runtime.get_task_instruction()

    def evaluate_reward(self):
        reward, test_output = self.env.runtime._calculate_reward(get_test_output = True)
        return reward
        
    def step(self, action: str):
        action = Action.from_string(action)
        # Check for max steps
        if self.total_steps > self.max_steps:
            return "Max Time steps", 0, True, {}

        if action.function_name == "":
            return "", 0, False, {}

        obs, reward, done, info = self.env.step(action, timeout = 20)
        if done:
            reward = self.evaluate_reward()

        self.total_steps += 1
        return str(obs), reward, done, {}

    def close(self):
        if self.env is not None:
            self.env.close()
     
class BatchSWEEnv(BatchedEnv):
    def __init__(
        self,
        batch_size,
    ):
        swe_dataset = load_dataset("r2e-edits/r2e-dockers-v1", split="train")
        self.envs = []
        self._env_id = []
        for i in range(batch_size):
            self.envs.append(SWEEnv(swe_dataset))
            self._env_id.append(f"{i}")

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
            obs = env.reset()
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
        def _close(idx):
            self.envs[idx].close()
            return True

        successes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            future_to_close = {
                executor.submit(_close, idx): idx for idx in range(len(self.envs))
            }
            for future in concurrent.futures.as_completed(future_to_close):
                success = future.result()
                
        try:
            os.system('docker stop $(docker ps -a -q)')
            os.system('docker rm $(docker ps -a -q)')
        except Exception as e:
            pass

    @staticmethod
    def from_extra_infos(extra_infos: List[Dict]) -> "BatchSWEEnv":
        return BatchSWEEnv(batch_size=len(extra_infos))
