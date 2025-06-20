import base64
import io
import logging
import re
import collections
from typing import List, Dict, Any, Tuple

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from PIL import Image

from rllm.agents.system_prompts import *
from rllm.agents.agent import BaseAgent, Step, Trajectory


logger = logging.getLogger(__name__)

def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class MiniWobAgent(BaseAgent):
    def __init__(self, chat_mode=False, use_html=True, use_axtree=True, use_screenshot=False, use_accumulate_thinking=True, cot_prompt=False, use_full_conversation=True, use_reward_shaping=False):
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=False,  # add visual effects
        )

        self.action_history = [] # all are in string

        # for interface compliance
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0
        self.reset()

        self.accumulate_thinking = use_accumulate_thinking
        self.cot_prompt = cot_prompt
        self.full_conversation = use_full_conversation
        self.reward_shaping = use_reward_shaping

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        obs = self._preproc_obs(observation)
        # Base message for the user
        user_prompt_content = self._format_msgs_as_str(self.get_user_msgs(obs))

        # initial state
        if not self.messages:
            self.messages.append(
                {
                    "role": "system", 
                    "content": self._format_msgs_as_str(self.get_system_msgs(obs))
                },
            )

        # Update the last step in the trajectory with the outcome (next_observation, reward, done, info)
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            # The observation received here is the 'next_observation' for the *previous* action/step
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        # Add the user message for the *next* interaction turn
        self.messages.append({
            "role": "user",
            "content": user_prompt_content
        })

        # Create a new step for the current state (with the observation that resulted from the last action)
        # This step's action, reward, etc., will be filled in by subsequent update_from_model and update_from_env calls
        if done:
            return
        
        cur_step = Step(
            observation=observation, 
            step=self.step
        )
        self._trajectory.steps.append(cur_step)
        
    def update_from_model(self, response: Any, **kwargs):
        if isinstance(response, str):
            content = response
        else: # OpenAI response
            try:
                content = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Failed to extract content from response: {response}. Error: {e}")
                content = str(response)

        thought = content
        if not self.accumulate_thinking:
            thought, sep, after = content.partition("</think>")
            if sep:
                content = after

        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        action_str = self._parse_model_response(content)

        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought # only thought if we aren't accumulating_thinking and extract </think> otherwise full response
        cur_step.action = action_str
        cur_step.model_response = content

        self.messages.append({"role": "assistant", "content": content})

        self.step += 1

        action_history_str = action_str if action_str != content else "Response is missing ``` ```"
        self.action_history.append(action_history_str)

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return self.messages
    
    @property
    def prompt(self) -> List[Dict[str, str]]:
        if self.full_conversation:
            return self.messages

        latest_msgs = [self.messages[0]] # system message
        has_assistant_msg = False
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                latest_msgs += self.messages[i + 1:] 
                has_assistant_msg = True
                break
        if not has_assistant_msg:
            latest_msgs += self.messages[1:]
        return latest_msgs

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.action_history = []
        self.step = 0

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            raise ValueError("get_current_state called before the first observation was processed.")
        return self._trajectory.steps[-1]


    def get_system_msgs(self, obs):
        system_msgs = []
        system_msgs.append({
            "type": "text",
            "text": self._get_system_prompt()
        })

        # Add goal information
        system_msgs.append({
            "type": "text",
            "text": "\n# Goal (Below is the goal you want to accomplish):\n\n"
        })
        system_msgs.extend(obs["goal_object"])  
        return system_msgs

    def get_user_msgs(self, user_obs):
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

        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# History of past actions
""",
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\
Action {i}:
{action}
""" if i != len(self.action_history) - 1 else f"""\
Last Action:
{action}
""",
                    } 
                    for i, action in enumerate(self.action_history)
                ]
            )

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

        # Add action space description
        user_msgs.append({
            "type": "text",
            "text": self._get_action_space_description()
        })

        # Add next action prompt
        user_msgs.append({
            "type": "text",
            "text": "# Next action\nThe task has not been completed yet. You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. The content must be in the same format as shown before in the Action Space. You can plan ahead but only 1 immediate action is needed."
        })

        return user_msgs
    

    def _preproc_obs(self, obs: dict) -> dict:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }


    def _get_system_prompt(self):
        return SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT

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
        if self.cot_prompt:
            return f"""\
# Action Space (This is the list of valid actions you are allowed to output after your chain-of-thought reasoning,
{self.action_set.describe(with_long_description=False, with_examples=False)}
Here are examples of actions with chain-of-thought reasoning:
Thought: I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
Action: ```click("12")```
Thought: I found the information requested by the user, I will send it to the chat.
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""
        else:
            return f"""\
# Action Space (This is the list of valid actions you are allowed to output,
{self.action_set.describe(with_long_description=False, with_examples=False)}
Here are examples of actions that can be returned:
Action: ```click("12")```
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""

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


    def _parse_model_response(self, response):
        """
        Extracts the last content enclosed within triple backticks (``` ```) from the response.

        If the response contains multiple segments wrapped in triple backticks, 
        this function returns the content of the **last** occurrence. 
        If no such formatting is found, it returns the entire response unmodified.

        Args:
            response (str): The raw text response to be processed.

        Returns:
            Tuple[str, str]: A tuple containing:
                - The extracted action (content from the last occurrence of triple backticks
                  or the full response if no match is found)
        """
        matches = re.findall(r'```(.*?)```', response, re.DOTALL)  # Find all occurrences
        if matches:
            return matches[-1]
        return response 

    def compute_training_reward(self, trajectory: Trajectory) -> float:
        if not trajectory:
            return 0

        reward = trajectory.steps[-1].reward
        if not self.reward_shaping:
            return reward
        
        reward_penalty = 0    
        for step in trajectory.steps:
            if not self.validate_step(step):
                reward_penalty = -0.5
                break
        return reward + reward_penalty

    def validate_step(self, trajectory_step):
        """
        Validates if the trajectory_step(dict) is valid or malformated.
        """
        thought = trajectory_step.thought
        action = trajectory_step.action

        # Thought and action are the same, meaning the parser didn't work
        if thought == action:
            return False

        # Response has action that results in error
        if trajectory_step["next_observation"]["last_action_error"]:
            return False
        
        return True

