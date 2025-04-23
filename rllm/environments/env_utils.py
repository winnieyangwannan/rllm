import numpy as np
import concurrent.futures
from contextlib import contextmanager
from typing import List, Dict, Any, Tuple, Callable, Iterator, Optional, Union


def add_trajectory_reward(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add trajectory reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.

    Returns:
        The updated trajectory with trajectory_reward added to each step.
    """
    if not trajectory:
        return trajectory
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory


def add_mc_return(trajectory: List[Dict[str, Any]], gamma: float = 0.95) -> List[Dict[str, Any]]:
    """
    Add Monte Carlo returns to each step in the trajectory.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.
        gamma: Discount factor for future rewards.

    Returns:
        The updated trajectory with mc_return added to each step.
    """
    if not trajectory:
        return trajectory
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1])) * gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1) / gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards * gamma_matrix, axis=1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


def add_training_reward(trajectory: List[Dict[str, Any]], training_reward: float) -> List[Dict[str, Any]]:
    """
    Add training reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.
        training_reward: The reward value to add to each step.

    Returns:
        The updated trajectory with training_reward added to each step.
    """
    if not trajectory:
        return trajectory
    for d in trajectory:
        d.update({"training_reward": training_reward})
    return trajectory


def compute_training_score(trajectory: List[Dict[str, Any]]) -> float:
    """
    Computes the reward for training in a given trajectory.

    Args:
        trajectory: A list of step dictionaries, where each step 
                   contains at least the key "training_reward".

    Returns:
        The training score extracted from the first step. Returns 0 if the 
        trajectory is empty.
    """
    return trajectory[0]["training_reward"] if trajectory else 0


def compute_environment_score(trajectory: List[Dict[str, Any]]) -> float:
    """
    Computes the overall environment score for a given trajectory.

    This function extracts the total trajectory reward from the first step in the trajectory.
    If the trajectory is empty, it returns a default score of 0.

    Args:
        trajectory: A list of step dictionaries, where the first step 
                   contains the key "trajectory_reward".

    Returns:
        The environment score extracted from the first step. Returns 0 if the 
        trajectory is empty.
    """
    return trajectory[0]["trajectory_reward"] if trajectory else 0


@contextmanager
def parallel_task_manager(
    func: Callable, 
    items: List[Any], 
    max_workers: int = 32
) -> Iterator[List[Tuple[int, Any]]]:
    """
    Execute a function in parallel for all items and collect results.
    
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
