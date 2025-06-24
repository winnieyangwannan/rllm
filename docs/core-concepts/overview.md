# Agent-Environment Orchestration

rLLM is built around several core components that work together to enable reinforcement learning for language models. This page provides a high-level overview of these components and how they interact.

rLLM consists of the following main components:

1. **Agents**: LLM-based agents that generate actions based on environment observations
2. **Environments**: Task-specific environments that provide observations and rewards
3. **AgentExecutionEngine**: Orchestrates interactions between agents and environments
4. **AgentTrainer**: RL algorithms to update agent policies based on rewards

## Agents

Agents are the core components in rLLM that generate intelligent actions based on environmental observations. They serve as the bridge between language models and interactive environments, enabling autonomous problem-solving and decision-making.

All agents inherit from the `BaseAgent` class, which defines the essential methods for environment interaction:

```python
from rllm.agents.agent import BaseAgent, Step, Trajectory

class BaseAgent(ABC):
    @abstractmethod
    def update_from_env(self, observation, reward, done, info, **kwargs):
        """Updates agent state after receiving environment feedback."""
        pass
        
    @abstractmethod
    def update_from_model(self, response, **kwargs):
        """Updates agent state after receiving model response."""
        pass
        
    @abstractmethod
    def reset(self):
        """Resets agent's internal state for new episode."""
        pass
        
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns messages formatted for chat completion."""
        return []
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns complete interaction history."""
        return Trajectory()

    def get_current_state(self) -> Step:
        """Return the most recent step."""
        assert self._trajectory.steps, "No active step available"
        return self._trajectory.steps[-1]
```

## Environments

Environments in rLLM define tasks and provide observations and rewards to agents. This page explains the environment architecture, available environment types, and how to create custom environments.

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

## Interaction Flow between Agent and Environment 

The agent-environment interaction follows this pattern:

1. **Initialization**: Agent calls `agent.reset()` to prepare for a new episode, environment calls `env.reset()` to provide initial observation.
2. **State Update**: Agent processes environment observation via `update_from_env()`
3. **Model Interaction**: Language model generates response using agent's `chat_completions`
4. **Response Processing**: Agent updates state via `update_from_model()`
5. **Repeat from Step 2**: Process repeats from Step 2 again until episode completion


## AgentExecutionEngine
 
The AgentExecutionEngine manages the interaction between agents and environments. It handles:

- **Agent-environment interaction**: Passing observations and actions
- **Async Parallel Rollout**: Running multiple agent-environment pairs simultaneously and asynchronously
- **Integration with training backend**: The agent execution engine handles trajectory rollout for RL integration