import base64
import io
import logging
import re
import collections

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from PIL import Image

from rllm.models.system_prompts import *
from rllm.models.agent import BaseAgent
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv


logger = logging.getLogger(__name__)


class FrozenLakeAgent(BaseAgent):

    SYSTEM_PROMPT = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action:
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action. The next Action MUST be wrapped in ``` ```.
"""

    def __init__(self):
        self.action_history = []

    def _pre_get_action(self, obs_act_seq):
        obs = obs_act_seq[0]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "Current Observation: \n" + obs},
        ]

        for step_idx, obs in enumerate(obs_act_seq[1:]):
            # 0 is assistant, 1 is user
            if step_idx % 2 == 1:
                messages.append({"role": "user", "content": "Current Observation: \n" + obs})
            else:
                assert obs != ""
                messages.append({"role": "assistant", "content": obs})

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
        DIRECTION_MAP = {"Left": 1, "Down": 2, "Right": 3, "Up": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, response.strip(), flags=re.IGNORECASE | re.X)
        
        if not match or not response:
            return str(FrozenLakeEnv.INVALID_ACTION)
        
        if match.group(2):   
            return str(int(match.group(2)))
        elif match.group(4): 
            return str(DIRECTION_MAP[match.group(4).capitalize()])
        elif match.group(5): 
            return str(int(match.group(5)))
        
        return str(FrozenLakeEnv.INVALID_ACTION)
    

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

        for traj_step in trajectory:
            if not self.validate_step(traj_step):
                return -1
            
        return trajectory[0]["trajectory_reward"]

    def validate_step(self, trajectory_step):
        """
        Validates if the trajectory_step(dict) is valid or malformated.
        """
        response = trajectory_step["response"]
        return True


    def convert_observation_to_string(self, obs, with_system_prompt=False):

        messages = ""
        if with_system_prompt:
            messages += self.SYSTEM_PROMPT

        messages += "Current Observation: \n" + obs

        return messages

