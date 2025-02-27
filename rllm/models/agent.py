from abc import ABC, abstractmethod
import openai
import time


class BaseAgent(ABC):

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        api_key=None,
        api_retries=3,
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.api_key = api_key
        self.api_retries = api_retries

        self.sampling_params = kwargs.get("sampling_params", None)

    @abstractmethod
    def _pre_get_action(self, obs):
        """
        Automatically called before get_action.
        Return in same format as OpenAI ChatInterface.
        """
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ]

    @abstractmethod
    def _post_get_action(self, response):
        """
        Automatically called after get_action.
        Return a string
        """
        return response

    @abstractmethod
    def update(self, action, observation, next_observation, reward, terminated, truncated, info):
        """
        Automatically called after taking an environment step during environment interation.
        No return required
        """
        return

    @abstractmethod
    def reset(self):
        return
    
    @abstractmethod
    def augment_reward(self, action, next_observation, reward):
        return reward
        