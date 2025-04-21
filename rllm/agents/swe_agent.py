import logging
import re
from typing import Tuple

import numpy as np

from rllm.agents.system_prompts import *
from rllm.agents.agent import BaseAgent

# from r2egym.agenthub.action import Action

# def parse_response(response_text: str) -> Tuple[str, Action]:
#     """
#     Extracts:
#     - thought: everything before the first <function=...> block
#     - action: the entire first <function=...></function> block
#     Returns (thought, action).
#     """
#     # Regex to match (non-greedily) from `<function=` up to the first `</function>`
#     pattern = re.compile(r"(?s)(<function=.*?</function>)")
#     match = pattern.search(response_text)

#     if match:
#         action = match.group(1)  # The entire <function=...></function> block
#         thought = response_text[: match.start()]  # Everything before the block
#     else:
#         # If no match, treat entire text as "thought"
#         thought = response_text
#         action = ""

#     # Strip leading/trailing whitespace
#     thought = thought.strip()
#     action = action.strip()

#     # convert action to Action object
#     action = Action.from_string(action)

#     return thought, action

logger = logging.getLogger(__name__)

class SWEAgent(BaseAgent):
    pass
#     def __init__(self):
#         self.action_history = [] # all are in string

#     def _pre_get_action(self, trajectory):
#         obs = trajectory[0]["next_observation"] # initial state

#         system_msgs = self.get_system_msg()

#         messages = [
#             {"role": "system", "content": self._format_msgs_as_str(system_msgs)},
#             {"role": "user", "content": self._format_msgs_as_str(self.get_user_msg(obs))},
#         ]
#         for i, step in enumerate(trajectory[1:]):
#             response = step["response"]
#             next_observation = step["next_observation"]

#             # response
#             messages.append({"role": "assistant", "content": response})

#             # next observation
#             obs = next_observation
#             usr_msg = self.get_user_msg(obs, first_obs=False)
#             messages.append({"role": "user", "content": self._format_msgs_as_str(usr_msg)})
           
#         return messages
    
#     def get_system_msg(self):
#         system_msgs = []
#         system_msgs.append({
#             "type": "text",
#             "text": self._get_system_prompt()
#         })
#         return system_msgs

#     def get_user_msg(self, user_obs, first_obs=True):
#         if not first_obs:
#             return [{"type": "text", "text": user_obs}]
#         user_msgs = []
#         user_msgs.append({
#             "type": "text",
#             "text": """
# Consider the following github issue:
# <github_issue>
# {problem_statement}
# </github_issue>

# Can you help me implement the necessary changes to the repository to fix the <github_issue>?
# I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
# Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

# IMPORTANT TIP:
# Follow these steps to resolve the issue:
# 1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
# 2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
# 3. Edit the sourcecode of the repo to resolve the issue
# 4. Rerun your reproduce script and confirm that the error is fixed!
# 5. Think about edgecases and make sure your fix handles them as well
# 6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
# 7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files. 
# """.format(problem_statement=user_obs)
#         })
#         return user_msgs

#     def _get_system_prompt(self):
#         return SYSTEM_SWE_PROMPT

#     def _format_msgs_as_str(self, msgs):
#         prompt_text_strings = []
#         for message in msgs:
#             prompt_text_strings.append(message["text"])
#         return " ".join(prompt_text_strings)

#     # def _post_get_action(self, response):
#     #     """
#     #     Extracts the last content enclosed within triple backticks (``` ```) from the response.

#     #     If the response contains multiple segments wrapped in triple backticks, 
#     #     this function returns the content of the **last** occurrence. 
#     #     If no such formatting is found, it returns the entire response unmodified.

#     #     Args:
#     #         response (str): The raw text response to be processed.

#     #     Returns:
#     #         str: The extracted content from the last occurrence of triple backticks, 
#     #             or the full response if no match is found.
#     #     """
#     #     # TODO: FIXME: Tianjun: we need to re-implement this function
#     #     thought, action = parse_response(response)
#     #     return action.to_xml_string()

#     def update(self, action, observation, next_observation, reward, terminated, truncated, info):
#         self.action_history.append(action)

#     def reset(self):
#         self.action_history = []

#     def compute_training_reward(self, trajectory):
#         """
#         Computes the training reward signal based on the entire trajectory.
#         """
#         if not trajectory:
#             return 0
        
#         if trajectory[0]["trajectory_reward"] == 1:
#             return 1

#         # for traj_step in trajectory:
#         #     if not self.validate_step(traj_step):
#         #         return -1
            
#         return 0

#     def convert_observation_to_string(self, obs, with_system_prompt=False):
#         messages = []
#         if with_system_prompt:
#             messages.extend(self.get_system_msg())

#         messages.extend(self.get_user_msg(obs, first_obs=with_system_prompt))

#         return self._format_msgs_as_str(messages)
