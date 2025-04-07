from abc import ABC, abstractmethod
from typing import List
from typing import Any, List, Tuple, Dict

class BaseEnv(ABC):

    @property
    @abstractmethod
    def env_id(self) -> str:
        """Return the unique environment IDs the instance represents. An environment ID should be unique to that specific type of environment"""
        pass

    @abstractmethod
    def reset(self, seed: int = 0) -> Tuple[List, List]:
        """Standard Gym reset method."""
        pass

    @abstractmethod
    def step(self, action):
        """Standard Gym step method."""
        pass