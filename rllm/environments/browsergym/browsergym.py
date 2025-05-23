import time
import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import Dict, List, Any, Tuple
import numpy as np
from typing import Dict, Any, List, Optional, Union
from rllm.environments.base.base_env import BaseEnv
from browser_pilot.entrypoint.client import CloudClient
from browser_pilot.entrypoint.env import CloudEnv
import traceback

USE_CLOUD_ENV = True


def with_retry(num_retries=3):
    max_wait = 16

    def decorator(func):
        def wrapper(*args, **kwargs):
            cur_wait = 1
            for i in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(
                        f"Error in {func.__name__}: {e}, retrying in {cur_wait} seconds. Traceback: {traceback.format_exc()}"
                    )
                    time.sleep(cur_wait)
                    cur_wait *= 2
                    cur_wait = min(cur_wait, max_wait)

            return func(*args, **kwargs)

        return wrapper

    return decorator


class BrowserGym(BaseEnv):
    def __init__(
        self,
        env_id: str = "browsergym/openended",
        task: Optional[Dict] = None,  # Optional tasks, used for openended
        client: Optional[CloudClient] = None,
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
        if USE_CLOUD_ENV:
            self.env = CloudEnv(
                url="ws://localhost:9999/send_and_wait",
                id=self._env_id,  # "browsergym_async/webarena.{id}"
                client=client,
                timeout=30000,
                slow_mo=1000,
            )
        else:
            self.env = gym.make(env_id, **worker_kwargs)

    @property
    def env_id(self):
        return self._env_id

    @with_retry(num_retries=5)
    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action, timeout=30000):
        obs, reward, terminated, truncated, extra_info = self.env.step(action, timeout)
        return obs, reward, terminated or truncated, extra_info

    @with_retry(num_retries=5)
    def close(self):
        self.env.close()

    @staticmethod
    def from_json(extra_info) -> "BrowserGym":
        return BrowserGym(env_id=extra_info["env_id"])

    @staticmethod
    def is_multithread_safe() -> bool:
        return True


if __name__ == "__main__":
    env = BrowserGym(env_id="539")
    obs = env.reset()
    print(obs)
    env.close()
