import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import Dict, List, Any, Tuple
import numpy as np
from typing import Dict, Any, List, Optional, Union
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


    @property
    def env_id(self) -> str:
        return self._env_id

    
    def reset(self, seed=0):
        return self.env.reset(seed)


    def step(self, action):
        return self.env.step(action)

    @staticmethod
    def from_extra_info(extra_info) -> "BrowserGym":
        return BrowserGym(env_id=extra_info["environment_id"])