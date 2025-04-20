import os
import concurrent.futures
from typing import List, Optional, Tuple, Any, Dict
import uuid
from contextlib import contextmanager

import numpy as np
from datasets import load_dataset, Dataset

import r2egym
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.action import Action

R2EGYM_PATH = os.path.dirname(r2egym.__file__)
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]
R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"

class SWEEnv:
    """Software Engineering Environment."""

    def __init__(self, dataset: Dataset = None, idx: int = None, max_steps: int = 40, timeout: int = 90, delete_image: bool = True):
        """Initialize the SWE environment.

        Args:
            dataset: Dataset containing the tasks
            idx: Index of the task to use
            max_steps: Maximum number of steps allowed
            timeout: Timeout for each step in seconds
        """
        if dataset is None:
            dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
        self.dataset = dataset
        if not idx:
            idx = np.random.randint(0, len(self.dataset))
        assert len(self.dataset) > idx and idx >= 0, "Select index out of range"
        self.idx = idx
        self.max_steps = max_steps
        self.timeout = timeout
        self.total_steps = 0
        self.delete_image = delete_image
        env_args = EnvArgs(ds=self.dataset[self.idx])
        self.env = RepoEnv(env_args)

    def reset(self):
        """Reset the environment."""
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

        if self.delete_image:
            docker_image = self.env.runtime.docker_image
            os.system(f"docker rmi {docker_image}")

if __name__ == "__main__":
    dataset = load_dataset("R2E-Gym/SWE-Bench-Lite", split="test")
    env = SWEEnv(dataset=dataset, idx=0)
    init_obs = env.reset()
    print(init_obs)
    env.close()