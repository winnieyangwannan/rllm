from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv


class MultiTurnEnvironment(BaseEnv, ABC):
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None,
                 max_turns: int = 3,
                 **kwargs):
        """
        Initialize the multi-turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "questions" field
                  with a list of questions for each turn
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__()
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        self.history = []
    
    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random
        if seed is not None:
            random.seed(seed)
        
        # Use the provided task if available, otherwise use the default task
        if task is not None:
            self.task = task
        
        self.done = False
        self.current_turn = 0
        self.history = []
        
        # Return the first question
        return {"question": self.task["question"]}, {}
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Response string from the LLM
            
        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)
        
        # Calculate reward for the current turn using the abstract method
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)
        
        # Increment turn counter
        self.current_turn += 1
        
        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task
        
        return next_obs, reward, self.done, self.task
    
    @abstractmethod
    def get_reward_and_next_obs(self, task: Dict, action: Any) -> Tuple[float, Dict]:
        """
        Abstract method to compute the reward based on the task and action.
        
        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent
            
        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        pass

    @staticmethod
    def from_json(info: Dict) -> "MultiTurnEnvironment":
        return MultiTurnEnvironment(
            task=info["task"],
            max_turns=info.get("max_turns", 3)
        )