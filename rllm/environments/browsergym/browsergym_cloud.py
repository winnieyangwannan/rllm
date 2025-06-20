import logging
import random
import time

from browser_pilot.entrypoint.client import CloudClient
from browser_pilot.entrypoint.env import CloudEnv

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


def with_retry(num_retries=3):
    max_wait = 16

    def decorator(func):
        def wrapper(*args, **kwargs):
            cur_wait = 1 + random.random()
            for i in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in {func.__name__}: {e}, retrying in {cur_wait} seconds.")
                    time.sleep(cur_wait)
                    cur_wait *= 1 + random.random()
                    cur_wait = min(cur_wait, max_wait)

            raise Exception(f"Failed to execute {func.__name__} after {num_retries} retries")

        return wrapper

    return decorator


class BrowserGymCloud(BaseEnv):
    def __init__(
        self,
        client: CloudClient = None,
        env_id=None,
        timeout=30000,
        slow_mo=1000,
        url="ws://localhost:9999/send_and_wait",
    ):
        if client is None:
            client = CloudClient(
                url=url,
                max_concurrency=10,
            )
        self.env = None
        self.client = client
        self.env_id = env_id
        self.timeout = timeout
        self.slow_mo = slow_mo
        self.url = url

    @with_retry(num_retries=5)
    def reset(self, task: dict = None):
        if task is None:
            task = {}
        if self.env is not None:
            self.env.close()
            self.env = None

        self.env = CloudEnv(
            url=self.url,
            id=self.env_id,  # "browsergym_async/webarena.{id}"
            client=self.client,
            timeout=self.timeout,
            slow_mo=self.slow_mo,
        )
        print(f"try reset url: {self.url}, {self.env_id}, {self.client}, {self.timeout}, {self.slow_mo}")
        return self.env.reset()

        # # clean up old env
        # task = task.copy()
        # env_id = task.pop("env_id", self.env_id)
        # assert env_id is not None, "env_id is required"
        # # task = task.pop("task", None)
        # env_kwargs = task.pop("env_kwargs", {})
        # timeout = task.pop("timeout", self.timeout)
        # slow_mo = task.pop("slow_mo", self.slow_mo)

        # if self.env is not None:
        #     self.env.close()
        #     self.env = None

        # logger.debug(f"Resetting env {env_id} with task {task} and env_kwargs {env_kwargs}")

        # self.env = CloudEnv(
        #     url=self.url,
        #     id=env_id,  # "browsergym_async/webarena.{id}"
        #     client=self.client,
        #     timeout=timeout,
        #     slow_mo=slow_mo,
        #     **env_kwargs
        # )

        # obs, info = self.env.reset()
        # return obs, info

    def step(self, action, timeout=30000):
        obs, reward, terminated, truncated, extra_info = self.env.step(action, timeout)
        return obs, reward, terminated or truncated, extra_info

    def close(self):
        self._close()
        self.env = None

    @with_retry(num_retries=5)
    def _close(self):
        logger.debug("Closing env")
        self.env.close()
        logger.debug("Successfully closed env")

    @staticmethod
    def from_json(extra_info_json=None) -> "BrowserGymCloud":
        if extra_info_json is None:
            extra_info_json = {}
        env_id = extra_info_json["env_id"]
        url = extra_info_json["url"]
        return BrowserGymCloud(env_id=env_id, url=url)

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
    logger.info("Reset done")
    env.close()
