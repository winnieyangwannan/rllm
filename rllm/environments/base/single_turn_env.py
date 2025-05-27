from typing import Callable, Dict, Optional, Tuple, Any

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.rl_reward import rllm_reward_fn

class SingleTurnEnvironment(MultiTurnEnvironment):
    """
    A simple environment for single-turn interactions with LLMs.
    This is a special case of MultiTurnEnvironment where max_turns=1.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None, 
                **kwargs):
        """
        Initialize the single turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "question" field
        """
        super().__init__(task=task, max_turns=1, **kwargs)
        self.reward_fn = rllm_reward_fn
    
    def get_reward_and_next_obs(self, task: Dict, action: Any) -> Tuple[float, Dict]:
        """
        Compute the reward based on the task and action.
        
        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent
            
        Returns:
            Tuple of (reward: float, next_observation: Dict)
        """
        reward_response = self.reward_fn(
            data_source=task["data_source"], 
            llm_solution=action, 
            ground_truth=task["ground_truth"]
        )
        return reward_response.reward, {}

    @staticmethod
    def from_json(info: Dict) -> "SingleTurnEnvironment":
        return SingleTurnEnvironment(task=info["task"])