from typing import Any, Dict, List

from rllm.agents.agent import BaseAgent, Step, Trajectory

class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """
    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.accumulate_thinking = accumulate_thinking
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """Process environment feedback and update internal state."""
        
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self.trajectory.steps:
            # Initial problem presentation
            assert isinstance(observation, dict) and 'question' in observation
            question = observation['question']
            formatted_observation = f'{question} {self.instruction}'
        else:
            # Follow-up correction prompt
            formatted_observation = (
                "Your previous answer may contain a mistake. "
                "Please review it carefully and answer again. "
                "Put your final answer within \\boxed{}."
            )

        # If there are previous steps, update the last step's outcome
        if self.trajectory.steps:
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
        self.trajectory.steps.append(cur_step)

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state based on the model's response.
        """        
        assert self.trajectory.steps, "Trajectory should not be empty when update_from_model is called."
        
        # Update the current step in the trajectory
        cur_step = self.get_current_state()
        cur_step.model_response = response
        cur_step.action = response

        if not self.accumulate_thinking:
            _, sep, after = response.partition("</think>")
            if sep:
                response = after

        self.messages.append({"role": "assistant", "content": response})
        
        self.step += 1

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return conversation history for model interaction."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
