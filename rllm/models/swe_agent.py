import base64
import io
import logging
import re
import collections

import numpy as np

from rllm.models.system_prompts import *
from rllm.models.agent import BaseAgent


logger = logging.getLogger(__name__)

class SWEAgent(BaseAgent):
    def __init__(self):
        self.action_history = [] # all are in string

    def _pre_get_action(self, trajectory):
        obs = trajectory[0]["next_observation"] # initial state

        system_msgs = self.get_system_msg(obs)

        messages = [
            {"role": "system", "content": self._format_msgs_as_str(system_msgs)},
            {"role": "user", "content": self._format_msgs_as_str(self.get_user_msg(obs))},
        ]
        for i, step in enumerate(trajectory[1:]):
            response = step["response"]
            next_observation = step["next_observation"]

            # response
            messages.append({"role": "assistant", "content": response})

            # next observation
            obs = self._preproc_obs(next_observation)
            usr_msg = self.get_user_msg(obs, append_action=False)
            messages.append({"role": "user", "content": self._format_msgs_as_str(usr_msg)})
           
        return messages
    
    def get_system_msg(self, obs):
        system_msgs = []
        system_msgs.append({
            "type": "text",
            "text": self._get_system_prompt()
        })

        return system_msgs

    def get_user_msg(self, user_obs, append_action=True):
        user_msgs = []
        # Add open tabs information
        user_msgs.extend(self._format_open_tabs(
            user_obs["open_pages_urls"],
            user_obs["open_pages_titles"],
            user_obs["active_page_index"]
        ))

        # Add page information based on settings
        if self.use_axtree:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page Accessibility Tree\n\n{user_obs['axtree_txt']}\n\n"
            })

        if self.use_html:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page DOM\n\n{user_obs['pruned_html']}\n\n"
            })

        if self.use_screenshot:
            user_msgs.extend(self._format_screenshot(user_obs["screenshot"]))

        if user_obs["last_action_error"]:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action

{user_obs["last_action_error"]}

""",
                }
            )

        if append_action:
            # Add action space description
            user_msgs.append({
                "type": "text",
                "text": self._get_action_space_description()
            })

            # TODO: check whether this should be part of all observation or not
            # Add next action prompt
            user_msgs.append({
                "type": "text",
                "text": "# Next action\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. MAKE SURE TO WRAP YOU FINAL ACTION in ```action``` YOU MUST PUT IN THIS EXACT STYLE FOR THE ACTION TO BE VALID. The content must be in the same format as shown before in the Action Space. Only 1 action is needed."
            })

        return user_msgs
    

    def _get_system_prompt(self):
        return SYSTEM_SWE_PROMPT


    def _format_open_tabs(self, urls: list, titles: list, active_index: int) -> list:
        messages = [{"type": "text", "text": "# Currently open tabs (This is the current active tabs)\n"}]

        for idx, (url, title) in enumerate(zip(urls, titles)):
            active_marker = " (active tab)" if idx == active_index else ""
            messages.append({
                "type": "text",
                "text": f"Tab {idx}{active_marker}\n  Title: {title}\n  URL: {url}\n"
            })
        return messages


    def _format_screenshot(self, screenshot: np.ndarray):
        messages = []
        messages.append(
                {
                    "type": "text",
                    "text": """\
# Current page Screenshot
""",
                }
            )
        messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(screenshot),
                    "detail": "auto",
                },  # Literal["low", "high", "auto"] = "auto"
            }
        )
        return messages


    def _get_action_space_description(self):
        return f"""\
# Action Space (This is the list of valid actions you are allowed to output after your chain-of-thought reasoning, YOU MUST OUTPUT EXACTLY IN THIS FORMAT FOR ACTION TO BE VALID)
{self.action_set.describe(with_long_description=False, with_examples=False)}
Here are examples of actions with chain-of-thought reasoning:
Thought: I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
Action: ```click("12")```
Thought: I found the information requested by the user, I will send it to the chat.
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""


    def _format_action_history(self, last_action_error):
        msgs = []
        msgs.append(
                {
                    "type": "text",
                    "text": """\
# History of past actions
""",
                }
            )
        msgs.extend(
            [
                {
                    "type": "text",
                    "text": f"""\
{action}
""",
                }
                for action in self.action_history
            ]
        )

        if last_action_error:
            msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action
{last_action_error}
""",
                }
            )
        return msgs


    def _format_msgs_as_str(self, msgs):
        prompt_text_strings = []
        for message in msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append(
                            "image_url: " + image_url[:30] + "... (truncated)"
                        )
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )
        return " ".join(prompt_text_strings)


    def _post_get_action(self, response):
        """
        Extracts the last content enclosed within triple backticks (``` ```) from the response.

        If the response contains multiple segments wrapped in triple backticks, 
        this function returns the content of the **last** occurrence. 
        If no such formatting is found, it returns the entire response unmodified.

        Args:
            response (str): The raw text response to be processed.

        Returns:
            str: The extracted content from the last occurrence of triple backticks, 
                or the full response if no match is found.
        """
        matches = re.findall(r'```(.*?)```', response, re.DOTALL)  # Find all occurrences
        if matches:
            return matches[-1] 
        return response 


    def update(self, action, observation, next_observation, reward, terminated, truncated, info):
        self.action_history.append(action)


    def reset(self):
        self.action_history = []


    def compute_training_reward(self, trajectory):
        """
        Computes the training reward signal based on the entire trajectory.
        """
        if not trajectory:
            return 0
        
        if trajectory[0]["trajectory_reward"] == 1:
            return 1

        # for traj_step in trajectory:
        #     if not self.validate_step(traj_step):
        #         return -1
            
        return 0

    def convert_observation_to_string(self, obs, with_system_prompt=False):
        obs = self._preproc_obs(obs)

        messages = []
        if with_system_prompt:
            messages.extend(self.get_system_msg(obs))

        messages.extend(self.get_user_msg(obs, append_action=with_system_prompt))

        return self._format_msgs_as_str(messages)
