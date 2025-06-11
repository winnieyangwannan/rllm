import logging
from typing import Any, Dict, List

from rllm.agents.agent import BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """
    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step and put the final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.accumulate_thinking = accumulate_thinking
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and 'question' in observation, "Initial observation must be a dict with a 'question' key."
            question = observation['question']
            formatted_observation = f'{question} {self.instruction}'
        else:
            formatted_observation = "Your previous answer may contain a mistake. Please review it carefully and answer again. Put your final answer within \\boxed{}."

        # If there are previous steps, update the last step's outcome
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = formatted_observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        
        if done:
            return
        
        self.messages.append({
            "role": "user",
            "content": formatted_observation
        })
        cur_step = Step(
            observation=formatted_observation,
            step=self.step
        )
        self._trajectory.steps.append(cur_step)

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state based on the model's response.
        """        
        assert self.trajectory.steps, "Trajectory should not be empty when update_from_model is called."
        
        # Update the current step in the trajectory
        cur_step = self.get_current_state()
        cur_step.model_response = response

        if not self.accumulate_thinking:
            _, sep, after = response.partition("</think>")
            if sep:
                response = after

        self.messages.append({"role": "assistant", "content": response})
        
        self.step += 1

    def reset(self):
        """
        Resets the agent's internal state for a new episode.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns the history of messages for chat completion."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory object."""
        return self._trajectory
