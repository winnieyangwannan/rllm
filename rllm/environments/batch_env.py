from abc import ABC, abstractmethod
from typing import List
from typing import Any, List, Tuple, Dict

class BatchedEnv(ABC):

    @property
    @abstractmethod
    def env_id(self) -> List[str]:
        """List of environment IDs the instance represents. An environment ID should be unique to that specific type of environment"""
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size for the instance. Should equal to len(env_id)"""
        pass

    @abstractmethod
    def reset(self, seed: int = 0) -> Tuple[List, List]:
        """
        Reset all environments in parallel.

        Args:
            seed (int, optional): Random seed for environment initialization. Default is 0.

        Returns:
            Tuple[List, List]: A tuple containing:
                - observations (List): Initial observations.
                - infos (List): Additional environment information.
        """
        pass

    @abstractmethod
    def step(self, actions: List[Any], env_idxs: List[int] = []) -> Tuple[List, List, List, List, List]:
        """
        Steps the selected environments in parallel. If no specific environment indices are provided (`env_idxs=[]`), all environments are stepped.

        Args:
            actions (List[Any]): A list of actions, one per selected environment.
            env_idxs (List[int], optional): A list of environment indices to step. 
                                            If empty, all environments are stepped.

        Returns:
            Tuple[List, List, List, List, List]: A tuple containing:
                - observations (List): The new observations after stepping the environments.
                - rewards (List): The rewards received for each environment.
                - terminateds (List): Boolean flags indicating if each environment has reached 
                                    a terminal state.
                - truncateds (List): Boolean flags indicating if each environment was truncated 
                                    due to a limit (e.g., max steps).
                - infos (List): Additional environment-specific information.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close all environments and terminate processes.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_json(extra_infos: List[Dict]) -> "BatchedEnv":
        """Abstract static method that constructs an instance from a list of dictionaries. Used for veRL training"""
        pass
