# Agents

Agents in rLLM are responsible for generating actions based on observations from environments. This page explains the agent architecture, available agent types, and how to create custom agents.

## Agent Architecture

All agents in rLLM inherit from the `BaseAgent` class, which defines the core interface for interaction with environments:

```python
from rllm.agents.agent import BaseAgent, Step, Trajectory

class BaseAgent(ABC):
    @abstractmethod
    def update_from_env(self, observation, reward, done, info, **kwargs):
        """Updates agent state after environment feedback."""
        pass
        
    @abstractmethod
    def update_from_model(self, response, **kwargs):
        """Updates agent state after model response."""
        pass
        
    @abstractmethod
    def reset(self):
        """Resets agent's internal state for new episode."""
        pass
        
    @abstractmethod
    def get_current_state(self) -> Step:
        """Returns current step/state of the agent."""
        pass
    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns messages for chat completion."""
        return []
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns agent's trajectory."""
        return Trajectory()
    
    @property
    def prompt(self) -> List[Dict[str, str]]:
        """Returns messages for next model prompt."""
        return self.chat_completions
```

### Key Concepts

- **Step**: Represents a single interaction step with observation, thought, action, and reward
- **Trajectory**: Collection of steps representing an episode's full interaction history
- **Chat Completions**: OpenAI-compatible message format for model interaction

The typical agent flow is:

1. Environment calls `update_from_env()` with observation and feedback
2. Agent processes observation and updates internal state
3. Model generates response based on agent's `chat_completions`
4. Agent calls `update_from_model()` to process model response into actions
5. Cycle repeats until episode completion

## Best Practices

1. **State Management**: Always update trajectory steps properly in both `update_from_env` and `update_from_model`

2. **Error Handling**: Implement robust parsing for model responses, as they may not always follow expected formats

3. **Resource Management**: Use `reset()` method to clean up state between episodes

4. **Observation Processing**: Format observations consistently for your model's requirements

5. **Action Parsing**: Implement reliable action extraction from model responses

6. **Testing**: Test agents with various environment configurations and edge cases

## Next Steps

- Learn about [Environments](environments.md) that agents interact with
- Explore the [API Reference](../api/agents.md) for detailed agent documentation
- See [Examples](../examples/agents.md) for complete agent implementations 