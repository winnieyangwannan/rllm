from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    observation: Any = None
    thought: str = ""
    action: Any = None
    reward: float = 0.0
    next_observation: Any = None
    done: bool = False
    # Store additional information from the environment or anything else.
    info: dict = field(default_factory=dict)
    step: int = 0
    model_response: str = ""
    mc_return: float = 0.0  # Monte Carlo estimate of returns.


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self):
        return {
            "steps": [asdict(step) for step in self.steps],
            "reward": float(self.reward),  # Convert numpy float to Python float
        }


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Converts agent's internal state into a list of OAI chat completions."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Converts agent's internal state into a Trajectory object."""
        return Trajectory()

    @abstractmethod
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation (Any): The observation after stepping through environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended due to termination.
            info (dict): Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Step:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        assert self.trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self.trajectory.steps[-1]
