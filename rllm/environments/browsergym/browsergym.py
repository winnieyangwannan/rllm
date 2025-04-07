import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import Dict, List, Any, Tuple
import numpy as np
from typing import Dict, Any, List, Optional, Union
from ..base_env import BaseEnv

# from ..batch_env import BatchedEnv
# def browser_gym_worker(connection, env_id: str, env_kwargs: Dict[str, Any]):
#     """
#     Generic worker function that initializes different BrowserGym environments with arbitrary arguments.
#     Args:
#         connection: Multiprocessing connection object for communication.
#         env_id: The environment ID.
#         env_kwargs: Arbitrary keyword arguments passed to `gym.make()`.
#     """
#     try:
#         env = gym.make(env_id, **env_kwargs)
#     except TypeError as e:
#         raise ValueError(f"Invalid arguments for environment {env_id}: {e}")

#     # Worker loop
#     while True:
#         try:
#             cmd, data = connection.recv()
#             if cmd == "reset":
#                 obs, info = env.reset(seed=data)
#                 connection.send((obs, info))
#             elif cmd == "step":
#                 obs, reward, terminated, truncated, info = env.step(data)
#                 connection.send((obs, reward, terminated, truncated, info))
#             elif cmd == "close":
#                 env.close()
#                 connection.close()
#                 break
#         except EOFError:
#             break

# class BatchBrowserGym(BatchedEnv):
#     def __init__(
#         self,
#         batch_size: int = 1,
#         env_id: Union[str, List[str]] = "browsergym/openended",
#         tasks: Optional[List[Dict]] = None,  # Optional list of tasks, used for openended
#         **env_kwargs,
#     ):
#         """
#         Initialize batched browser gym environment using multiple processes
        
#         Args:
#             batch_size: Number of parallel environments
#             env_id: Gym environment ID to use
#         """
#         if tasks:
#             assert len(tasks) == batch_size, "Number of task_kwargs must match batch_size"

#         if isinstance(env_id, str):
#             env_id = [env_id] * batch_size  # Duplicate for all workers if env_id is a string
#         else:
#             assert len(env_id) == batch_size, "Number of env_id entries must match batch size"

#         self._batch_size = batch_size
#         self._env_id = env_id
#         self.tasks = tasks
#         self.env_kwargs = env_kwargs

#         self.processes = []
#         self.connections = []

#         # Create multiple processes with pipe connections
#         for i in range(self.batch_size):
#             parent_conn, child_conn = Pipe()

#             worker_kwargs = env_kwargs.copy()
#             if tasks:
#                 worker_kwargs["task_kwargs"] = tasks[i] 

#             process = Process(
#                 target=browser_gym_worker,
#                 args=(child_conn, env_id[i], worker_kwargs)
#             )

#             process.start()
#             self.processes.append(process)
#             self.connections.append(parent_conn)

#     @property
#     def env_id(self) -> List[str]:
#         return self._env_id

#     @property
#     def batch_size(self) -> int:
#         return self._batch_size
    
#     def reset(self, seed=0) -> Tuple[List, List]:
#         """Reset all environments in parallel"""
#         # Send reset command to all workers
#         for conn in self.connections:
#             conn.send(("reset", seed))
#         # Collect results
#         results = [conn.recv() for conn in self.connections]
#         observations, infos = zip(*results)

#         return list(observations), list(infos)


#     def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:
#         """
#         Steps the selected environments in parallel. If no specific environment indices are provided (`env_idxs=[]`), all environments are stepped.

#         Args:
#             actions (List[Any]): A list of actions, one per selected environment.
#             env_idxs (List[int], optional): A list of environment indices to step. 
#                                             If empty, all environments are stepped.

#         Returns:
#             Tuple[List, List, List, List, List]: A tuple containing:
#                 - observations (List): The new observations after stepping the environments.
#                 - rewards (List): The rewards received for each environment.
#                 - terminateds (List): Boolean flags indicating if each environment has reached 
#                                     a terminal state.
#                 - truncateds (List): Boolean flags indicating if each environment was truncated 
#                                     due to a limit (e.g., max steps).
#                 - infos (List): Additional environment-specific information.

#         Raises:
#             AssertionError: If the number of actions does not match the batch size when stepping all environments.
#             AssertionError: If the number of actions does not match the number of specified environment indices.

#         """

#         if not env_idxs:
#             assert len(actions) == self.batch_size, "Number of actions must match batch size"
#             env_idxs = list(range(len(actions)))

#         assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"


#         # Send step command with actions
#         for i, env_idx in enumerate(env_idxs):
#             self.connections[env_idx].send(("step", actions[i]))

#         # Collect results
#         results = [self.connections[i].recv() for i in env_idxs]
#         observations, rewards, terminateds, truncateds, infos = zip(*results)

#         return (list(observations), list(rewards), list(terminateds), 
#                 list(truncateds), list(infos))


#     def close(self):
#         """Close all environments and terminate processes"""
#         # Send close command to all workers
#         for conn in self.connections:
#             conn.send(("close", None))

#         # Wait for all processes to finish
#         for process in self.processes:
#             process.join()
#             process.terminate()

#         # Close all connections
#         for conn in self.connections:
#             conn.close()

#     @staticmethod
#     def from_extra_infos(extra_infos: List[Dict]) -> "BatchBrowserGym":
#         env_ids = [
#             i["environment_id"] for i in extra_infos
#         ]

#         return BatchBrowserGym(batch_size=len(env_ids), env_id=env_ids)
    



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