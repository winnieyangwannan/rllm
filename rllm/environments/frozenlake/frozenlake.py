"""
run `pip install "gymnasium[toy-text]"` to install gymnasium
"""

"""
Adapted from nice codes from gymnasium.envs.toy_text.frozen_lake.generate_random_map
Modify it so that the start and end points are random
"""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding
from typing import List, Optional, Tuple, Any, Dict
import hashlib
import numpy as np
import copy
from ..batch_env import BatchedEnv


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    # generate random start and end points

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])

        while True:
            start_r = np_random.integers(0, size)
            start_c = np_random.integers(0, size)
            goal_r = np_random.integers(0, size)
            goal_c = np_random.integers(0, size)
            
            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break
            
        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"
        
        valid = is_valid(board, size)
    return ["".join(x) for x in board]




class FrozenLakeEnv(GymFrozenLakeEnv):
    """
    Inherits from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv

    ## Description
    The game starts with the player at random location of the frozen lake grid world with the
    goal located at another random location for the 4x4 environment.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
    - 0: Still
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Starting State
    The episode starts with the player at random location

    ## Rewards
    NOTE added -0.1 as penalty for invalid action
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Arguments
    `is_slippery`: if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3

    ## Example
    P   _   _   _
    _   _   _   O
    O   _   O   _
    O   _   _   G
    """

    # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,
        b"F": 1,
        b"H": 2,
        b"G": 3,
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        3: "Right",
        4: "Up",
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1


    def __init__(self, **kwargs):

        desc = kwargs.pop('desc', None)
        is_slippery = kwargs.pop('is_slippery', True)
        size = kwargs.pop('size', 8)
        p = kwargs.pop('p', 0.8)
        seed = kwargs.pop('seed', None)
        if desc is None:
            random_map = generate_random_map(size=size, p=p, seed=seed)
        else:
            random_map = np.asarray(copy.deepcopy(desc), dtype="c")

        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=is_slippery
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        
        
        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": copy.deepcopy(desc),
            "seed": seed,
        }
        self.action_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        } # map from custom Env action to action defined in FrozenLakeEnv in gymnasium

        self.reward = 0
        self._valid_actions = []

    def _get_player_position(self):
        return (self.s // self.ncol, self.s % self.ncol) # (row, col)

    def reset(
            self,
            mode='tiny_rgb_array',
            reset_map=True,
            seed=None
    ):
        """
        Reset the environment, there are two options:
        1. reset the map, generate a new map (reset_map=True)
        2. reset the environment with the same map, while putting the agent back to the start position (reset_map=False)
        Both can reset the seed
        NOTE if seed is the same, the map will be the same
        """
        
        if reset_map:
            self.__init__(
                size=self.map_kwargs["size"],
                p=self.map_kwargs["p"],
                seed=seed,
                is_slippery=self.env_kwargs["is_slippery"],
            )
        GymFrozenLakeEnv.reset(self, seed=seed)
        return self.render(mode)
    
    def finished(self):
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"GH"
        
    def success(self):
        """
        Check if the agent has reached the goal (G) or hole (H)
        """
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"G"
    
    def step(self, action: str):
        """
        - Map custom action to gymnasium FrozenLakeEnv action and take the step
        - Check if the action is effective (whether player moves in the env).
        """
        if self.success():
            return self.render(), 0, True, {"action_is_effective": False}
        
        if not action:
            action = self.INVALID_ACTION
        action = int(action)
        assert isinstance(action, int), "Action must be an integer"
        assert not self.success(), "Agent has already reached the goal or hole"

        if action == self.INVALID_ACTION: # no penalty for invalid action
            return self.render(), 0, False, {"action_is_effective": False}
        
        prev_player_position = int(self.s)

        player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(self, self.action_map[action])

        obs = self.render()
        return obs, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}
    

     
    def render(self, mode='tiny_rgb_array'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array', 'ansi']
        if mode in ['rgb_array', 'ansi']:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b'S')
        room_state[position_S] = b'F'

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b'P'

        if mode == 'state':
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b'H':
                room_state[position_P] = 4
            elif self.desc[position_P] == b'G':
                room_state[position_P] = 5
            return room_state
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)


class BatchFrozenLakeEnv(BatchedEnv):
    def __init__(
        self,
        batch_size,
        seeds, 
        sizes,
        ps,
    ):
        self.envs = []
        self._env_id = []
        for i in range(batch_size):
            seed = seeds[i]
            size = sizes[i]
            p = ps[i]
            self.envs.append(FrozenLakeEnv(size=size, seed=seed, p=p))
            self._env_id.append(f"{seed}-{size}-{p}")

        self._batch_size = batch_size

    @property
    def env_id(self) -> List[str]:
        return self._env_id

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    def reset(self, seed=0) -> Tuple[List, List]:
        observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset(reset_map=False)
            observations.append(obs)
        return observations, [{}] * self.batch_size


    def step(self, actions: List[Any], env_idxs: List[int]=[]) -> Tuple[List, List, List, List, List]:

        if not env_idxs:
            assert len(actions) == self.batch_size, "Number of actions must match batch size"
            env_idxs = list(range(len(actions)))

        assert len(actions) == len(env_idxs), f"Number of actions ({len(actions)}) must match the env used {len(env_idxs)}"

        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        # Send step command with actions
        for i, env_idx in enumerate(env_idxs):
            obs, reward, done, info = self.envs[env_idx].step(actions[i])
            observations.append(obs),
            rewards.append(reward)
            terminateds.append(done)
            truncateds.append(False)
            infos.append(info)


        return (observations, rewards, terminateds, 
                truncateds, infos)


    def close(self):
        return

    @staticmethod
    def from_extra_infos(extra_infos: List[Dict]) -> "BatchFrozenLakeEnv":
        seeds = [
            i["seed"] for i in extra_infos
        ]
        sizes = [
            i["size"] for i in extra_infos
        ]
        ps = [
            i["p"] for i in extra_infos
        ]

        return BatchFrozenLakeEnv(batch_size=len(seeds), seeds=seeds, sizes=sizes, ps=ps)

