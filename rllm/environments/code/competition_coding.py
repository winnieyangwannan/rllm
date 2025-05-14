from typing import Dict, Optional, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.rl_reward import rllm_reward_fn

class CompetitionCodingEnv(MultiTurnEnvironment):
    """
    Environment for competitive coding tasks that inherits from MultiTurnEnvironment.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None,
                 max_turns: int = 3,
                 **kwargs):
        """
        Initialize the competitive coding environment.
        
        Args:
            task: Dictionary containing the task information
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.reward_fn = rllm_reward_fn
    
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
            data_source=task["data_source"], 
            llm_solution=action, 
            ground_truth=task["ground_truth"]
        )        
        return reward_response.reward, reward_response.metadata
    
    @staticmethod
    def from_json(info: Dict) -> "CompetitionCodingEnv":
        return CompetitionCodingEnv(
            task=info["task"],
            max_turns=info.get("max_turns", 2)
        )