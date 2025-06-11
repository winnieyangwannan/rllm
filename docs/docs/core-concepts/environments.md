# Environments and Rewards

Environments in rLLM define tasks and provide observations and rewards to agents. This page explains the environment architecture, available environment types, and how to create custom environments.

## Environment Architecture

All environments in rLLM inherit from the `BaseEnv` class, which follows the Gymnasium interface with rLLM-specific extensions:

```python
from rllm.environments.base.base_env import BaseEnv

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Reset environment and return initial observation and info."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action and return (observation, reward, done, info)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass

    @staticmethod
    @abstractmethod
    def from_dict(env_args: Dict) -> "BaseEnv":
        """Create environment instance from dictionary. This function is used both during inference and training to instantiate a new environment instance, so it has to be implemented properly."""
        pass
```

## Reward
rLLM provides reward function for competition math and coding, which we use to train DeepScaleR and DeepCoder. Our SingleTurnEnvironment supports passing in your custom reward function, as long as it satisfies the following interface:

```python
@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""
    
    def __call__(self, task_info: Dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.
        
        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution
            
        Returns:
            RewardOutput: The calculated reward value as a RewardOutput object
        """
        ...
```

where 

```python
@dataclass(slots=True, kw_only=True)
class RewardOutput:
    """Data structure for the output of reward calculations.

    Attributes:
        reward (float): The computed reward value based on the evaluation of the model's response.
        metadata (dict): Additional information about the reward calculation.
        is_correct (bool): A boolean flag indicating whether the model's response is deemed correct.
    """
    reward: float
    metadata: dict = field(default_factory=dict)
    is_correct: Optional[bool] = None
```


## Implementing your custom environment with custom reward function
```python
from rllm import 
class MultiTurnMathEnv(BaseEnv)
```


## Next Steps

- Learn about [AgentExecutionEngine](rewards.md) for ru
- Explore the [API Reference](../api/environments.md) for detailed environment documentation