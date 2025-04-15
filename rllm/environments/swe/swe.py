import os
import concurrent.futures
from typing import List, Optional, Tuple, Any, Dict
import uuid
from contextlib import contextmanager

import numpy as np
from datasets import load_dataset, Dataset
from gymnasium.utils import seeding

import r2egym
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.action import Action

from rllm.environments.batch_env import BatchedEnv

R2EGYM_PATH = os.path.dirname(r2egym.__file__)
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search_dir.py"),
]
R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]


@contextmanager
def parallel_task_manager(func, items, max_workers=32):
    """Execute a function in parallel for all items and collect results.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers
        
    Yields:
        List of (idx, result) tuples
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(func, *item): i for i, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            result = future.result()
            results.append((idx, result))
    yield results


class SWEEnv:
    """Software Engineering Environment."""

    def __init__(self, dataset: Dataset, idx: int = None, max_steps: int = 40, timeout: int = 90):
        """Initialize the SWE environment.

        Args:
            dataset: Dataset containing the tasks
            idx: Index of the task to use
            max_steps: Maximum number of steps allowed
            timeout: Timeout for each step in seconds
        """
        self.dataset = dataset
        if not idx:
            idx = np.random.randint(0, len(self.dataset))
        assert len(self.dataset) > idx, "Select index out of range"
        self.idx = idx
        self.max_steps = max_steps
        self.timeout = timeout
        self.env = None
        self.total_steps = 0
    
    def reset(self):
        """Reset the environment."""
        env_args = EnvArgs(ds=self.dataset[self.idx])
        self.env = RepoEnv(env_args)

        # Reset environment and docker runtime.
        self.env.reset()
        self.env.runtime.reset()
        self.env.runtime.setup_env()
        self.env.add_commands(R2EGYM_COMMAND_FILES)
        self.total_steps = 0

        # Polls docker runtime to get task instruction.
        return self.env.get_task_instruction()

    def compute_reward(self):
        """Compute the reward for the current state."""
        reward, test_output = self.env.runtime._calculate_reward(get_test_output=True)
        return reward
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        action: Action = Action.from_string(action)
        # Check for max steps
        if self.total_steps > self.max_steps:
            return "Max steps exceeded.", 0, True, {}

        if not action.function_name:
            return "", 0, False, {}

        # RepoEnv always return 0 reward, must be evaluated by DockerRuntime.
        obs, reward, done, info = self.env.step(action, timeout=self.timeout)
        if done:
            reward = self.compute_reward()

        self.total_steps += 1
        return str(obs), reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class BatchSWEEnv(BatchedEnv):
    """Batched Software Engineering Environment."""
    
    def __init__(
        self,
        batch_size,
        dataset_name: str = "R2E-Gym/R2E-Gym-Lite",
        split: str = "train", # Must be either "train" or "test"
        seeds: List[int] = None,
        truncate_dataset_size: int = None, # Limit # of Docker images to use
    ):
        """Initialize the batched SWE environment.
        
        Args:
            batch_size: Number of environments to run in parallel
            seeds: Random seeds for each environment
        """
        self._batch_size = batch_size
        assert dataset_name in R2E_ENV_IDS, \
            f"Dataset name {dataset_name} not in {R2E_ENV_IDS}"
        self.dataset_name = dataset_name

        assert split in ["train", "test"], \
            f"Split {split} must be either 'train' or 'test'"
        self.split = split

        self.seeds = seeds
        
        self.full_dataset = load_dataset(dataset_name, split=split)
        if truncate_dataset_size:
            self.full_dataset = self.full_dataset.select(range(truncate_dataset_size))
        
        self.envs = []
        self.env_ids = []
        for i in range(batch_size):
            if self.seeds:
                np_random, _ = seeding.np_random(seeds[i])
                select_idx = np_random.integers(0, len(self.full_dataset))
            else:
                select_idx = None
            # Ensure idx is a standard Python int for dataset indexing
            env_idx = int(select_idx) if select_idx is not None else None
            self.envs.append(SWEEnv(self.full_dataset, idx=env_idx))
            self.env_ids.append(f"SWE:{select_idx}-{str(uuid.uuid4())[:6]}")

        self.max_workers = 32

    @property
    def env_id(self) -> List[str]:
        """Get the environment IDs."""
        return self.env_ids

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size

    def reset(self, seed=0) -> Tuple[List, List]:
        """Reset all environments in parallel.
        
        Args:
            seed: Random seed
            
        Returns:
            Tuple of (observations, infos)
        """
        def _reset_env(idx):
            obs = self.envs[idx].reset()
            return idx, obs

        with parallel_task_manager(
            _reset_env,
            [(idx,) for idx in range(len(self.envs))],
            max_workers=self.max_workers
        ) as results:
            observations = [obs for _, (_, obs) in sorted(results, key=lambda x: x[0])]

        return observations, [{}] * self.batch_size

    def step(self, actions: List[str], env_idxs: List[int] = []) -> Tuple[List, List, List, List, List]:
        """Step the environments in parallel.
        
        Args:
            actions: List of actions to take
            env_idxs: List of environment indices to step
            
        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"

        observations, rewards, dones, truncates, infos = [], [], [], [], []

        def _step_env(idx, action):
            obs, reward, done, info = self.envs[idx].step(action)
            return idx, obs, reward, done, info

        with parallel_task_manager(
            _step_env,
            [(idx, action) for idx, action in zip(env_idxs, actions)],
            max_workers=self.max_workers
        ) as results:
            # Unpack and reorder results
            sorted_results = sorted(results, key=lambda x: x[0])
            for _, (_, obs, reward, done, info) in sorted_results:
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                truncates.append(False)
                infos.append(info)

        return observations, rewards, dones, truncates, infos

    def close(self):
        """Close all environments."""

        def _close_env(idx):
            self.envs[idx].close()

        with parallel_task_manager(
            _close_env,
            [(idx,) for idx in range(len(self.envs))],
            max_workers=self.max_workers
        ) as _:
            pass

        try:
            # Stop all running containers
            os.system('docker stop $(docker ps -a -q)')
            # Remove all containers from memory
            os.system('docker rm $(docker ps -a -q)')
        except Exception:
            pass

    @staticmethod
    def from_json(extra_infos: List[Dict]) -> "BatchSWEEnv":
        """Create a BatchSWEEnv from extra infos.
        
        Args:
            extra_infos: List of extra info dictionaries
            
        Returns:
            BatchSWEEnv instance
        """
        seeds = [
            i["seed"] for i in extra_infos
        ]
        # Pass default dataset_name and split, as they are not in extra_infos
        return BatchSWEEnv(
            batch_size=len(extra_infos), 
            seeds=seeds,
            dataset_name="R2E-Gym/SWE-Bench-Lite", # Add default
            split="test" # Add default
        )

if __name__ == "__main__":
    env = BatchSWEEnv(batch_size=2)
    print(env.reset())
    print(env.close())
