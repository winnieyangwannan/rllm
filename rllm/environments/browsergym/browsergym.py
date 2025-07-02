from typing import Dict, Optional
import multiprocessing as mp
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

# Compatible with async.io
class BrowserGymProcess(BaseEnv):
    def __init__(self, env_id="browsergym/openended", task=None, **env_kwargs):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(
            target=self._worker, args=(self.child_conn, env_id, task, env_kwargs)
        )
        self.process.start()

    def _worker(self, conn, env_id, task, env_kwargs):
        env = gym.make(env_id, task_kwargs=task, **env_kwargs) if task else gym.make(env_id, **env_kwargs)
        try:
            while True:
                cmd, data = conn.recv()
                if cmd == "reset":
                    obs = env.reset()
                    conn.send(obs)
                elif cmd == "step":
                    action = data
                    obs, reward, terminated, truncated, extra_info = env.step(action)
                    conn.send((obs, reward, terminated or truncated, extra_info))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    break
        except EOFError:
            env.close()

    def reset(self):
        self.parent_conn.send(("reset", None))
        return self.parent_conn.recv()

    def step(self, action):
        self.parent_conn.send(("step", action))
        return self.parent_conn.recv()

    def close(self):
        self.parent_conn.send(("close", None))
        self.process.join()

    @staticmethod
    def from_json(extra_info) -> "BrowserGymProcess":
        return BrowserGymProcess(env_id=extra_info["env_id"])

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
