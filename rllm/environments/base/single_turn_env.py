from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.reward_fn import RewardFunction, zero_reward


class SingleTurnEnvironment(MultiTurnEnvironment):
    """
    A simple environment for single-turn interactions with LLMs.
    This is a special case of MultiTurnEnvironment where max_turns=1.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None, 
                 reward_fn: Optional[RewardFunction] = None,
                 **kwargs):
        """
        Initialize the single turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "question" field
        """
        super().__init__(task=task, max_turns=1, **kwargs)
        self.reward_fn = reward_fn or zero_reward
    
    def get_reward_and_next_obs(self, task: Dict, action: Any) -> Tuple[float, Dict]:
        """
        Compute the reward based on the task and action.
        
        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent
            
        Returns:
            Tuple of (reward: float, next_observation: Dict)
        """
        reward_output = self.reward_fn(
            task_info=task,
            action=action
        )
        
        return reward_output.reward, {}

    @staticmethod
    def from_dict(env_info: Dict) -> "SingleTurnEnvironment":
        reward_fn = env_info.pop('reward_fn', None)
        if 'task' in env_info:
            task = env_info['task']
        else:
            task = env_info
        return SingleTurnEnvironment(task=extra_info)