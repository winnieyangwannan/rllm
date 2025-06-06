import json
from typing import Dict, Protocol, runtime_checkable

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.rewards.code_reward import rllm_reward_fn_code
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.reward_types import RewardOutput


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""
    
    def __call__(self, task_info: Dict, action: str) -> float | RewardOutput:
        """
        Calculate the reward for an agent's action.
        
        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution
            
        Returns:
            float | RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...

# Simple example implementation
def zero_reward(task_info: Dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.
    
    Args:
        task: The task dictionary
        action: The agent's response
        
    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})


def code_reward_fn(task_info: Dict, action: str) -> RewardOutput:
    """
    A reward function for code tasks that implements the RewardFunction protocol.
    
    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution
        
    Returns:
        float: The calculated reward value based on code execution results
    """
    data_source = task_info.get("data_source", "")
    ground_truth = task_info.get("ground_truth", {})
    
    # Pass through to the existing code reward function
    reward_output = rllm_reward_fn_code(data_source, action, ground_truth)
    return reward_output


def math_reward_fn(task_info: Dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.
    
    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution
        
    Returns:
        float: The calculated reward value based on math evaluation
    """
    data_source = task_info.get("data_source", "")
    ground_truth = task_info.get("ground_truth", "")
    extra_info = task_info.get("extra_info", {})
    
    # Pass through to the existing math reward function
    reward_output = rllm_reward_fn_math(data_source, action, ground_truth, extra_info)
    return reward_output


def rllm_reward_fn(task_info: Dict, action: str) -> RewardOutput:
    """
    A reward function that handles both code and math tasks.
    
    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution
        
    Returns:
        float: The calculated reward value
    """
    data_source = task_info.get("data_source", "")

    if data_source.upper() in [e.name for e in list(TestDataset.Code) + list(TrainDataset.Code)]:
        # For code tasks, ensure ground_truth is properly parsed
        ground_truth = task_info.get("ground_truth", "")
        try:
            if isinstance(ground_truth, str):
                task_info["ground_truth"] = json.loads(ground_truth)
        except json.JSONDecodeError:
            return RewardOutput(reward=0.0, metadata={"error": "Invalid ground truth format"})
        
        return code_reward_fn(task_info, action)
    else:
        return math_reward_fn(task_info, action)