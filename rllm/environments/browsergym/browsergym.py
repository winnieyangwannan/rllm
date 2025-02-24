import gymnasium as gym
from multiprocessing import Process, Pipe
from typing import Dict, List, Any, Tuple
import numpy as np
from typing import Dict, Any, List, Optional, Union

def browser_gym_worker(connection, env_id: str, env_kwargs: Dict[str, Any]):
    """
    Generic worker function that initializes different BrowserGym environments with arbitrary arguments.
    Args:
        connection: Multiprocessing connection object for communication.
        env_id: The environment ID.
        env_kwargs: Arbitrary keyword arguments passed to `gym.make()`.
    """
    try:
        env = gym.make(env_id, **env_kwargs)
    except TypeError as e:
        raise ValueError(f"Invalid arguments for environment {env_id}: {e}")

    # Worker loop
    while True:
        try:
            cmd, data = connection.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                connection.send((obs, info))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                connection.send((obs, reward, terminated, truncated, info))
            elif cmd == "close":
                env.close()
                connection.close()
                break
        except EOFError:
            break

class BatchBrowserGym:
    def __init__(
        self,
        batch_size: int = 1,
        env_id: Union[str, List[str]] = "browsergym/openended",
        tasks: Optional[List[Dict]] = None,  # Optional list of tasks, used for openended
        **env_kwargs,
    ):
        """
        Initialize batched browser gym environment using multiple processes
        
        Args:
            batch_size: Number of parallel environments
            env_id: Gym environment ID to use
        """
        if tasks:
            assert len(tasks) == batch_size, "Number of task_kwargs must match batch_size"

        if isinstance(env_id, str):
            env_id = [env_id] * batch_size  # Duplicate for all workers if env_id is a string
        else:
            assert len(env_id) == batch_size, "Number of env_id entries must match batch size"

        self.batch_size = batch_size
        self.env_id = env_id
        self.tasks = tasks
        self.env_kwargs = env_kwargs

        self.processes = []
        self.connections = []

        # Create multiple processes with pipe connections
        for i in range(self.batch_size):
            parent_conn, child_conn = Pipe()

            worker_kwargs = env_kwargs.copy()
            if tasks:
                worker_kwargs["task_kwargs"] = tasks[i] 

            process = Process(
                target=browser_gym_worker,
                args=(child_conn, env_id[i], worker_kwargs)
            )

            process.start()
            self.processes.append(process)
            self.connections.append(parent_conn)


    def reset(self, seed=0) -> Tuple[List, List]:
        """Reset all environments in parallel"""
        # Send reset command to all workers
        for conn in self.connections:
            conn.send(("reset", seed))
        # Collect results
        results = [conn.recv() for conn in self.connections]
        observations, infos = zip(*results)

        return list(observations), list(infos)


    def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:
        """
        Step environments in parallel. If env_idx is [], then all environment is used.
        
        Args:
            actions: List of actions for each environment
            env_idx: List of environment indexs used. If [], then all environment is used.
        """

        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == env_idxs, "Number of actions must match the env used"


        # Send step command with actions
        for i in env_idxs:
            self.connections[i].send("step", actions[i])

        # Collect results
        results = [self.connections[i].recv() for i in env_idxs]
        observations, rewards, terminateds, truncateds, infos = zip(*results)

        return (list(observations), list(rewards), list(terminateds), 
                list(truncateds), list(infos))


    def close(self):
        """Close all environments and terminate processes"""
        # Send close command to all workers
        for conn in self.connections:
            conn.send(("close", None))

        # Wait for all processes to finish
        for process in self.processes:
            process.join()
            process.terminate()

        # Close all connections
        for conn in self.connections:
            conn.close()
