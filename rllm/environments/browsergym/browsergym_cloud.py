import logging
import time
import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import Dict, List, Any, Tuple
import numpy as np
from typing import Dict, Any, List, Optional, Union
# from rllm.environments.base.base_env import BaseEnv
from browser_pilot.entrypoint.client import CloudClient
from browser_pilot.entrypoint.env import CloudEnv
import random

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# client = CloudClient(url="ws://localhost:9999/send_and_wait", max_concurrency=128)


def with_retry(num_retries=3):
    max_wait = 16

    def decorator(func):
        def wrapper(*args, **kwargs):
            cur_wait = 1 + random.random()
            for i in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Error in {func.__name__}: {e}, retrying in {cur_wait} seconds."
                    )
                    time.sleep(cur_wait)
                    cur_wait *= 1 + random.random()
                    cur_wait = min(cur_wait, max_wait)

            raise Exception(f"Failed to execute {func.__name__} after {num_retries} retries")

        return wrapper

    return decorator


class BrowserGymCloud:
    def __init__(
        self,
        client: CloudClient = None,
    ):
        """
        Initialize batched browser gym environment using multiple processes
        Args:
            batch_size: Number of parallel environments
            env_id: Gym environment ID to use
        """
        self.env = None
        self.client = client

    @with_retry(num_retries=5)
    def reset(self, task: Dict = {}):
        # clean up old env
        task = task.copy()
        env_id = task.pop("env_id", None)
        assert env_id is not None, "env_id is required"
        # task = task.pop("task", None)
        env_kwargs = task.pop("env_kwargs", {})
        timeout = task.pop("timeout", 30000)
        slow_mo = task.pop("slow_mo", 1000)

        if self.env is not None:
            self.env.close()
            self.env = None

        logger.debug(f"Resetting env {env_id} with task {task} and env_kwargs {env_kwargs}")
        
        self.env = CloudEnv(
            url="ws://localhost:9999/send_and_wait",
            id=env_id,  # "browsergym_async/webarena.{id}"
            client=self.client,
            timeout=timeout,
            slow_mo=slow_mo,
            **env_kwargs
        )

        obs, info = self.env.reset()
        return obs, info

    def step(self, action, timeout=30000):
        obs, reward, terminated, truncated, extra_info = self.env.step(action, timeout)
        return obs, reward, terminated or truncated, extra_info

    def close(self):
        self._close()
        self.env = None

    @with_retry(num_retries=5)
    def _close(self):
        logger.debug(f"Closing env")
        self.env.close()
        logger.debug(f"Successfully closed env")

    @staticmethod
    def from_json() -> "BrowserGymCloud":
        return BrowserGymCloud()

    @staticmethod
    def is_multithread_safe() -> bool:
        return True


if __name__ == "__main__":
    extra_info = {
        "env_id": "browsergym_async/webarena.539",
        # "task": None,
        "env_kwargs": {},
        "timeout": 30000,
        "slow_mo": 1000,
    }
    env = BrowserGymCloud.from_json()
    obs = env.reset(task=extra_info)
    print(obs)
    logger.info(f"Reset done")
    env.close()
