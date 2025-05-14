import json
import logging
from typing import Dict, List, Any, Tuple

from rllm.agents.agent import BaseAgent, Step, Trajectory
from rllm.rewards.math_reward import rllm_reward_fn_math

logger = logging.getLogger(__name__)

class CodeAgent(BaseAgent):    
    """
    A code agent that iteratively writes code to solve a problem.
    """
    def __init__(self):
        """
        Initialize the MathAgent.
        """
        self.revise_instruction = "Here's the feedback from the previous attempt. Revise the code to fix the errors and improve the solution."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.max_tests = 4

    def format_test_results(self, test_results: List[Dict]) -> str:
        all_passed = True
        formatted_test_results = "Here are the results on the public test cases:\n"
        for i, test in enumerate(test_results[:self.max_tests]):
            if test["passed"]:
                formatted_test_results += f"### Test {i} passed\n"
            else:
                formatted_test_results += f"### Test {i+1} failed\n"
                all_passed = False
            
            # Add the input, expected, and passed fields
            formatted_test_results += f"  Input: {test['input']}\n"
            formatted_test_results += f"  Expected: {test['expected']}\n"
            # formatted_test_results += f"  Actual: {test['actual']}\n\n"

        if all_passed:
            formatted_test_results += "Congratulations! You've successfully passed all the public test cases. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."
        else:
            formatted_test_results += "Some test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly."

        return formatted_test_results
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback

        formatted_observation = None
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and 'question' in observation, "Initial observation must be a dict with a 'question' key."
            question = observation['question']
            formatted_observation = f'{question}'
        else:
            if "test_results" in observation:
                test_results = observation["test_results"]
                formatted_observation = self.format_test_results(test_results)

        if formatted_observation is not None:
            self.messages.append({
                "role": "user",
                "content": formatted_observation
            })
                # Create a new step for the current state
            cur_step = Step(
                observation=formatted_observation,
                step=self.step,
                reward=reward,
                done=done,
                info=info
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
