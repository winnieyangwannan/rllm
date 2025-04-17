import json
import logging
from typing import Dict, List

from rllm.agents.agent import BaseAgent
from rllm.rewards.math_reward import rllm_reward_fn_math

logger = logging.getLogger(__name__)

class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step.
    """
    def __init__(self):
        """
        Initialize the MathAgent.
        
        Args:
            model_name: The name of the model to use
        """
        self.instruction = "Let's think step by step and put the final answer within \\boxed{}."
        
        self.trajectory = []
        # no system prompt
        self.messages = []
        
    def _pre_get_action(self, trajectory: List[Dict]):
        self.messages.extend(self.format_observation_as_messages(trajectory[-1]['next_observation']))
        return self.messages
    
    def _post_get_action(self, response):
        if isinstance(response, str):
            self.messages.append({"role": "assistant", "content": response})
            return response
        else:
            completion = response
            content = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return content
    
    def update(self, action, observation, next_observation, reward, terminated, truncated, info):
        # Store the step information in the trajectory
        step = {
            "observation": observation,
            "next_observation": next_observation,
            "reward": reward,
            "action": action,
            "response": action,  # In this agent, the action is the response
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        self.trajectory.append(step)
    
    def reset(self):
        # Reset the agent's state
        self.trajectory = []
        self.messages = []
    
    def compute_training_reward(self, trajectory):
        return rllm_reward_fn_math('', trajectory[-1]['response'], trajectory[-1]['info']['answer'])
    
    def convert_observation_to_string(self, obs, with_system_prompt=False):
        if 'question' in obs:
            self.messages.append({"role": "user", "content": obs['question']})
        
        return obs
    
    def format_observation_as_messages(self, obs, **kwargs):
        messages = []
        if 'question' in obs:
            messages.append({"role": "user", "content": obs['question'] + self.instruction})
        
        return messages