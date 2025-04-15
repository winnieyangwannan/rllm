import base64
import io
import logging
import re
import collections

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from PIL import Image

from rllm.agents.system_prompts import *
from rllm.agents.agent import BaseAgent
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv


logger = logging.getLogger(__name__)


class FrozenLakeAgent(BaseAgent):

    SYSTEM_PROMPT = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```.
You should plan ahead and try to achieve it in minimum number of steps.

Below are examples for an interaction:
Example1:
User: Current Observation:
P   _   _   _   _
O   _   _   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the top right corner. It should reach G at the bottom right corner. I should move it closer to it. I can move right or down but there is a hole in down position and I can not move diagonally. There is no hole in my next movement right so I can move to right. Action: ```Right```

Example2:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   P
O   _   _   G
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the near G. It should reach G to its bottom. I should move to be on it. There is no hole in my next movement so I can move to down. Action: ```Down```

Example3:
User: Current Observation:
_   _   _   O   _
O   _   P   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right relative to P. I want to move closer so I should move right or down. But there is a hole at each position and I do not want to fall into holes. Up and left are both valid but left brings me closer. Action: ```Left```

Example4:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   O
O   G   P   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now near G. But game has not finished. P is not at G and I should never output invalid action. I need to recheck my understanding. P is not actually on G yet because they are not overlapping, it needs reach G to its left. Action: ```Left```

Example5:
User: Current Observation:
_   _   _   O   _
O   _   P   _   _
O   _   O   O   O
O   _   O   G   _
O   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right corner of P. I can move left, right, or up. Move right will initially bring me closer but I can't reach G that way. Move up and left means I can still reach G. Move up will result in 9 steps in total while left is 7 steps. I need to move left. Action: ```Left```

Now it is your turn, please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    def __init__(self):
        self.action_history = []

    def _pre_get_action(self, trajectory):
        obs = trajectory[0]["next_observation"]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "Current Observation: \n" + obs + "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."},
        ]
        last_obs = obs
        for i, step in enumerate(trajectory[1:]):
            response = step["response"]
            next_observation = step["next_observation"]

            # response
            messages.append({"role": "assistant", "content": response})
        
            # next observation
            user_msg = "Current Observation: \n" + next_observation + "\n " + "You have not achieved the goal, P has not reached G yet. Please give the next action."
            if last_obs == next_observation:
                user_msg += "Your last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response."
            last_obs = next_observation
            messages.append({"role": "user", "content": user_msg})
            
        return messages
    

    def _post_get_action(self, response):
        """
        Extract action from text.
        NOTE: the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
        - 0: Still (Invalid Action)
        - 1: Left
        - 2: Down
        - 3: Right
        - 4: Up
        """

        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}
    
        # Extract the last content enclosed in triple backticks
        matches = re.findall(r'```(.*?)```', response, re.DOTALL)
        if not matches:
            return str(FrozenLakeEnv.INVALID_ACTION)  # No valid action found
        
        extracted_text = matches[-1].strip().lower()  # Take the last match and normalize case

        # Try to match it to a valid action
        if extracted_text in DIRECTION_MAP:
            return str(DIRECTION_MAP[extracted_text])  # Return mapped number
        elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
            return str(int(extracted_text))  # If it's a valid number, return as is

        return str(FrozenLakeEnv.INVALID_ACTION)  # If nothing matches, return invalid action

        # DIRECTION_MAP = {"Left": 1, "Down": 2, "Right": 3, "Up": 4}
        # # Remove <|im_end|> if it exists
        # response = response.replace("<|im_end|>", "").strip()
        # response = response.lower()
        # # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        # pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        # match = re.fullmatch(pattern, response.strip(), flags=re.IGNORECASE | re.X)
        
        # if not match or not response:
        #     return str(FrozenLakeEnv.INVALID_ACTION)
        
        # if match.group(2):   
        #     return str(int(match.group(2)))
        # elif match.group(4): 
        #     return str(DIRECTION_MAP[match.group(4).capitalize()])
        # elif match.group(5): 
        #     return str(int(match.group(5)))
        
        # return str(FrozenLakeEnv.INVALID_ACTION)
    

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

        for traj_step in trajectory:
            if not self.validate_step(traj_step):
                return -1
            
        return 0

    def validate_step(self, trajectory_step):
        """
        Validates if the trajectory_step(dict) is valid or malformated.
        """
        response = trajectory_step["response"]
        if self._post_get_action(response) == str(FrozenLakeEnv.INVALID_ACTION):
            return False
        return True


    def format_observation_as_messages(self, obs, with_system_prompt=False):
        messages = []
        if with_system_prompt:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        messages.append({"role": "user", "content": "Current Observation: \n" + obs + "\n You have not achieved the goal, P has not reached G yet. Please give the next action."})

        return messages
