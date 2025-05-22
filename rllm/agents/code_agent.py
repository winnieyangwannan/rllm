import logging
from typing import Dict, List, Any, Tuple

from rllm.agents.agent import BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

class CompetitionCodingAgent(BaseAgent):    
    """
    A code agent that iteratively writes code to solve a problem.
    """
    def __init__(self, remove_thinking=True):
        """
        Initialize the MathAgent.
        """
        self.revise_instruction = "Here's the feedback from the previous attempt. Revise the code to fix the errors and improve the solution."
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.max_tests = 2
        self.remove_thinking = remove_thinking

    def format_test_results(self, test_results: List[Dict]) -> str:
        all_passed = True
        formatted_test_results = "Here are the results on the public test cases:\n"
        n_failed = 0
        for i, test in enumerate(test_results):
            if not test['passed']:
                n_failed += 1
                formatted_test_results += f"### Test {i+1} failed\n"
                all_passed = False

                # Add the input, expected, and passed fields
                formatted_test_results += f"  Input: {test['input']}\n" if len(test['input']) < 200 else ""
                formatted_test_results += f"  Expected: {test['expected']}\n" if len(test['expected']) < 200 else ""
                formatted_test_results += f"  Actual: {test['output']}\n\n" if 'output' in test and test['output'] is not None and len(test['output']) < 200 else ""
                formatted_test_results += f"  Error message: {test['error_message']}\n" if 'error_message' in test and test['error_message'] is not None and len(test['error_message']) < 200 else ""

                if n_failed > self.max_tests:
                    break

        if all_passed:
            formatted_test_results += "Congratulations! You've successfully passed all the public test cases. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."
        else:
            formatted_test_results += "Some test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code."

        return formatted_test_results
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback

        formatted_observation = ''
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and 'question' in observation, "Initial observation must be a dict with a 'question' key."
            question = observation['question']
            formatted_observation = f'{question}'
        else:
            if "test_results" in observation:
                test_results = observation["test_results"]
                formatted_observation = self.format_test_results(test_results)
            if 'error' in observation:
                formatted_observation = observation['error']

        # Update reward on the latest step
        if self.trajectory.steps:
            cur_step = self.trajectory.steps[-1]
            cur_step.reward = reward
            cur_step.step = self.step
            cur_step.done = done
            cur_step.info = info

        if done:
            return
        
        self.messages.append({
            "role": "user",
            "content": formatted_observation
        })
            # Create a new step for the current state
        new_step = Step(
            observation=formatted_observation,
            step=self.step
        )
        self._trajectory.steps.append(new_step)

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
        cur_step.thought = content 
        cur_step.action = content  
        cur_step.model_response = content

        # Remove <think></think> blocks from assistant messages
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