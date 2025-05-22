import json
import logging
from typing import Dict, List, Any, Tuple, Union

from rllm.agents.agent import BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CompetitionCodingAgent(BaseAgent):    
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
        self.max_tests = 2
    
    def format_test_results(self, test_results: List[Dict]) -> str:

        public_tests = []
        for i, test in enumerate(test_results):
            if "input" in test and isinstance(test["input"], str):
                strings_to_match = test["input"].split("\n")
            elif "input" in test and isinstance(test["input"], list):
                strings_to_match = test["input"]
            else:
                continue
            
            question = self.trajectory.steps[0].observation
            if all(s in question for s in strings_to_match):
                public_tests.append(test)

        if len(public_tests) == 0:
            print("Warning: No public tests found")
        else:
            print(f"Warning: Found {len(public_tests)} public tests")
        
        if len(public_tests) == 0 or all(test["passed"] for test in public_tests):
            return "Congratulations! You've successfully passed all the public test cases. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."
        
        else:
            formatted_test_results = ""
            n_failed = 0
            for i, test in enumerate(public_tests):
                if not test["passed"]:
                    formatted_test_results += f"### Test {i+1} failed\n"
                    formatted_test_results += f"  Input: {truncatefn(test['input'])}\n"
                    formatted_test_results += f"  Expected: {truncatefn(test['expected'])}\n"
                    formatted_test_results += f"  Actual: {truncatefn(test['output'])}\n\n" if 'output' in test and test['output'] is not None else ""
                    formatted_test_results += f"  Error message: {truncatefn(test['error_message'])}\n" if 'error_message' in test and test['error_message'] is not None else ""

                    n_failed += 1
                    if n_failed >= self.max_tests:
                        break

            return f"Here are the results on the public test cases:\n{formatted_test_results}\nSome test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code."
        
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
        # if "</think>" in content:
        #     think_start = content.find("<think>")
        #     think_end = content.find("</think>") + len("</think>")
        #     if think_start != -1 and think_end != -1 and think_end > think_start:
        #         # Remove full <think>...</think> block
        #         think_end += len("</think>")
        #         content = content[:think_start] + content[think_end:]
        #     elif think_end != -1:
        #         # Remove everything before and including </think>
        #         think_end += len("</think>")
        #         content = content[think_end:]


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