from abc import ABC, abstractmethod
from typing import List
from typing import Any, List, Tuple, Dict
import uuid

class BaseEnv(ABC):
    def __init__(self):
        self._env_id = str(uuid.uuid4())[:8] # Convert UUID to string before slicing
        
    @property
    def env_id(self) -> str:
        """Return the unique environment ID this instance represents. An environment ID should be unique to that specific type of environment"""
        return self._env_id
    
    @env_id.setter 
    def env_id(self, value: str):
        """Set the environment ID.
        
        Args:
            value: String ID to set for this environment
        """
        self._env_id = value

    @abstractmethod
    def reset(self, seed: int = 0, **kwargs) -> Tuple[List, List]:
        """Standard Gym reset method."""
        pass

    @abstractmethod
    def step(self, action):
        """Standard Gym step method."""
        pass

    def close(self):
        """Standard Gym close method."""
        pass

    @staticmethod
    def from_json(info: Dict) -> "BaseEnv":
        return BaseEnv()