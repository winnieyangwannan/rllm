from typing import Dict, Optional, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.rl_reward import rllm_reward_fn

class CompetitionCodingEnv(MultiTurnEnvironment):
    """
    Environment for competitive coding tasks that inherits from MultiTurnEnvironment.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None,
                 max_turns: int = 2,
                 reward_bonus_coeff: float = 0.0,
                 **kwargs):
        """
        Initialize the competitive coding environment.
        
        Args:
            task: Dictionary containing the task information
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.reward_fn = rllm_reward_fn
        self.prev_reward = None
        self.reward_bonus_coeff = reward_bonus_coeff

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
        self.prev_reward = None
        
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
        raw_reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Reward shaping
        if self.prev_reward is None:
            reward = raw_reward
        else:
            bonus = self.reward_bonus_coeff * (raw_reward - self.prev_reward)
            reward = raw_reward + bonus

        self.prev_reward = raw_reward
        
        # Increment turn counter
        self.current_turn += 1
        
        # Check if we've reached the maximum number of turns
        # if self.current_turn >= self.max_turns or reward == 1:
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task
        
        return next_obs, reward, self.done, self.task
    
    def get_reward_and_next_obs(self, task: Dict, action: str) -> Tuple[float, Dict]:
        """
        Compute the reward for a competitive coding task.
        
        Args:
            task: The task dictionary containing relevant information
            action: The response string from the LLM
            
        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        reward_response = self.reward_fn(
            data_source=task.get("data_source", ""), 
            llm_solution=action, 
            ground_truth=task["ground_truth"]
        )
        # all_passed_bonus = 1.0 if reward_response.metadata["all_passed"] else 0.0
        # n_passed_tests = reward_response.metadata["passed_tests"]
        # n_total_tests = reward_response.metadata["total_tests"]
        # partial_reward = n_passed_tests / n_total_tests
        return reward_response.reward, reward_response.metadata
    
    @staticmethod
    def from_dict(env_args: Dict) -> "CompetitionCodingEnv":
        return CompetitionCodingEnv(
            task=env_args["task"],
            max_turns=env_args.get("max_turns", 2),
            reward_bonus_coeff=env_args.get("reward_bonus_coeff", 0.0)
        )