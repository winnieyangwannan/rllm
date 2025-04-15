from typing import Callable, Dict, Optional

from rllm.environments.base.base_env import BaseEnv

class SingleTurnEnvironment(BaseEnv):
    """
    A simple environment for single-turn interactions with LLMs.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None, 
                 reward_fn: Optional[Callable[[str, Dict], float]] = None, **kwargs):
        """
        Initialize the single turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "question" field
            reward_fn: Function that takes (response, task) and returns a float reward
        """
        self.task = task
        self.reward_fn = reward_fn or (lambda response, task: 0.0)
        self.done = False
    
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
        
        # Calculate reward using the provided reward function
        reward = self.reward_fn(action, self.task)
        
        # Always terminate after a single step
        terminated = True
        truncated = False
        
        # Empty observation since we're done
        next_obs = {}

        info = self.task

        # Return results
        return next_obs, reward, terminated, truncated, info 