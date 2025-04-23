from typing import Callable, Dict, Optional

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.rl_reward import rllm_reward_fn

class SingleTurnEnvironment(BaseEnv):
    """
    A simple environment for single-turn interactions with LLMs.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None, 
                **kwargs):
        """
        Initialize the single turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "question" field
            reward_fn: Function that takes (response, task) and returns a float reward
        """
        super().__init__()
        self.task = task
        self.reward_fn = rllm_reward_fn
        self.done = False
        self._env_id = hash(str(self.task)) if self.task else ""
    
    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random
        if seed is not None:
            random.seed(seed)
        
        # Use the provided task if available, otherwise use the default task
        if task is not None:
            self.task = task
        
        self.done = False
        
        # Return a single observation in a list to maintain the batch structure
        return {"question": self.task["question"]}, {}
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Response string from the LLM
            
        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # In a single turn environment, any action leads to termination
        self.done = True
        reward = self.reward_fn(data_source="", llm_solution=action, ground_truth=self.task["answer"])
        # Return results
        return {}, reward, self.done, self.task

    @staticmethod
    def from_json(info: Dict) -> "SingleTurnEnvironment":
        return SingleTurnEnvironment(task=info["task"])