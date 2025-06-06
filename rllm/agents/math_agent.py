import logging
from typing import Any, Dict, List

from rllm.agents.agent import BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """
    def __init__(self, remove_thinking=False):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step and put the final answer within \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.remove_thinking = remove_thinking
        
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

    def update_from_model(self, response: Any, **kwargs):
        """
        Updates the agent's internal state based on the model's response.
        """
        # Extract content from the response
        if isinstance(response, str):
            content = response
        else:
            # Assuming response object similar to OAI completion
            content = response.choices[0].message.content
        
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."
        
        # Update the current step in the trajectory
        cur_step = self._trajectory.steps[-1]
        # For MathAgent, the model response represents both the thought and action.
        cur_step.thought = content 
        cur_step.action = content  # Or potentially parse out the boxed answer? For now, use full content.
        cur_step.model_response = content

        if self.remove_thinking:
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                # Remove full <think>...</think> block
                think_end += len("</think>")
                content = content[:think_start] + content[think_end:]
            elif think_end != -1:
                # Remove everything before and including </think>
                think_end += len("</think>")
                content = content[think_end:]

        # Add the assistant's response to the messages
        self.messages.append({"role": "assistant", "content": content})
        
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

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
