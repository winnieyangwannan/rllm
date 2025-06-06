from typing import Dict, Optional

import gymnasium as gym

from rllm.environments.base.base_env import BaseEnv


class BrowserGym(BaseEnv):
    def __init__(
        self,
        env_id: str = "browsergym/openended",
        task: Optional[Dict] = None,  # Optional tasks, used for openended
        **env_kwargs,
    ):
        """
        Initialize batched browser gym environment using multiple processes
        
        Args:
            batch_size: Number of parallel environments
            env_id: Gym environment ID to use
        """
       
        self._env_id = env_id
        self.task = task
        self.env_kwargs = env_kwargs
        worker_kwargs = env_kwargs.copy()
        if task:
            worker_kwargs["task_kwargs"] = task
        self.env = gym.make(env_id, **worker_kwargs)
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, extra_info = self.env.step(action)
        return obs, reward, terminated or truncated, extra_info
    
    def close(self):
        self.env.close()

    @staticmethod
    def from_json(extra_info) -> "BrowserGym":
        return BrowserGym(env_id=extra_info["env_id"])

    @staticmethod
    def is_multithread_safe() -> bool:
        return False