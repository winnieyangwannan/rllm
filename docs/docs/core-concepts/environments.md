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

## Environment Types

rLLM provides two main environment categories:

### Base Environment Classes

#### SingleTurnEnvironment

For tasks requiring only one interaction, such as question answering or classification:

```python
from rllm.environments import SingleTurnEnvironment

env = SingleTurnEnvironment(
    task={"question": "What is 12 * 12?", "ground_truth": "144"},
    reward_fn=my_reward_function  # Optional custom reward function
)

observation, info = env.reset()
# observation: {"question": "What is 12 * 12?"}

action = "The answer is 144."
next_obs, reward, terminated, truncated, info = env.step(action)
# terminated: True (single turn complete)
```

#### MultiTurnEnvironment

Abstract base class for multi-step interactions requiring custom reward logic:

```python
from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

class MyMultiTurnEnv(MultiTurnEnvironment):
    def get_reward_and_next_obs(self, task, action):
        # Custom reward computation
        reward = compute_my_reward(task, action)
        next_obs = generate_next_observation(task, action)
        return reward, next_obs

env = MyMultiTurnEnv(
    task={"question": "Let's solve a complex problem step by step."},
    max_turns=5
)
```

## Domain-Specific Environments

### ToolEnvironment

For training agents to use external tools and APIs with function calling:

```python
from rllm.environments import ToolEnvironment

env = ToolEnvironment(
    task={"question": "What is the population of France divided by the area of Texas?"},
    tools=["calculator", "web_search"],  # Available tools
    max_steps=10,
    reward_fn=my_reward_function
)

observation, info = env.reset()
# observation: {"question": "What is the population of France divided by the area of Texas?"}

# Agent makes tool calls
tool_calls = [{
    "id": "call_1",
    "function": {"name": "web_search", "arguments": '{"query": "population of France"}'}
}]
next_obs, reward, done, info = env.step(tool_calls)
# next_obs: {"tool_outputs": {"call_1": "France population is 67 million"}}
```

**Features:**
- Multi-tool support with threaded execution
- Tool call parsing and output formatting
- Configurable step limits and timeouts
- Support for finish actions to end episodes early

### BrowserGym

For web browsing and interaction training using the BrowserGym library:

```python
from rllm.environments import BrowserGym

env = BrowserGym(
    env_id="browsergym/openended",  # BrowserGym environment ID
    task={"instruction": "Find information about AI on Wikipedia"},
    # Additional BrowserGym kwargs
)

observation, info = env.reset()
action = "click [123]"  # BrowserGym action format
next_obs, reward, done, info = env.step(action)
```

**Features:**
- Integration with BrowserGym ecosystem
- Support for various web browsing tasks
- Customizable task configurations
- **Note**: Not multithread safe due to browser instances

### CompetitionCodingEnv

For competitive programming and code generation training with test feedback:

```python
from rllm.environments import CompetitionCodingEnv

env = CompetitionCodingEnv(
    task={
        "question": "Write a function to find the maximum subarray sum.",
        "ground_truth": test_cases_and_solutions
    },
    max_turns=2,  # Allow code revision
    reward_bonus_coeff=0.1  # Reward shaping coefficient
)

observation, info = env.reset()
code_solution = "def max_subarray(arr): ..."
next_obs, reward, done, info = env.step(code_solution)
# reward based on test case results
```

**Features:**
- Multi-turn code refinement
- Test case execution and feedback
- Reward shaping for iterative improvement
- Integration with code evaluation systems

### SWEEnv

Software engineering environment for GitHub issue resolution and repository interaction:

```python
from rllm.environments import SWEEnv

env = SWEEnv(
    entry=github_issue_data,  # Or use dataset with idx
    step_timeout=90,
    reward_timeout=300,
    backend="kubernetes",  # or "docker"
    verbose=True
)

observation, info = env.reset()
# observation: GitHub issue description and repository state

action = "<function=file_editor><parameter=command>view</parameter>..."
next_obs, reward, done, info = env.step(action)
# Executes action in repository environment
```

**Features:**
- Real repository environments using Docker/Kubernetes
- Integration with R2E-Gym and SWE-Bench datasets
- File editing, search, and bash execution capabilities
- Automatic patch evaluation for code changes

### FrozenLakeEnv

Grid-world reinforcement learning environment with configurable layouts:

```python
from rllm.environments import FrozenLakeEnv

env = FrozenLakeEnv(
    size=8,           # Grid size
    p=0.8,           # Probability of frozen tiles
    is_slippery=False, # Deterministic movement
    max_steps=5,     # Episode length limit
    seed=42
)

observation, info = env.reset()
action = "Right"  # or 1, 2, 3, 4 for Left, Down, Right, Up
next_obs, reward, done, info = env.step(action)
```

**Features:**
- Random map generation with guaranteed solvability
- Configurable grid sizes and layouts
- Text-based observations for LLM training
- Gymnasium compatibility with rLLM extensions

## Creating Custom Environments

You can create custom environments by inheriting from `BaseEnv` or its subclasses:

### Basic Custom Environment

```python
from rllm.environments.base.base_env import BaseEnv
from typing import Any, Dict, Tuple

class MyCustomEnvironment(BaseEnv):
    def __init__(self, task: Dict = None, max_steps: int = 10, **kwargs):
        super().__init__()
        self.task = task or {}
        self.max_steps = max_steps
        self.current_step = 0
        
    def reset(self, seed: int = 0, **kwargs) -> Tuple[Dict, Dict]:
        """Reset environment and return initial observation."""
        import random
        if seed:
            random.seed(seed)
            
        self.current_step = 0
        
        observation = {
            "instruction": self.task.get("instruction", "Hello! Let's begin."),
            "context": self.task.get("context", {}),
        }
        
        info = {"step": self.current_step, "max_steps": self.max_steps}
        return observation, info
        
    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return results."""
        self.current_step += 1
        
        # Process action and create observation
        observation = self._process_action(action)
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination conditions
        terminated = self._is_task_complete(action)
        truncated = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            "step": self.current_step,
            "action": action,
            "task_complete": terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: Any) -> Dict:
        """Process action and generate next observation."""
        return {"response": f"You said: {action}"}
    
    def _compute_reward(self, action: Any) -> float:
        """Compute reward for the action."""
        return 1.0 if "correct" in str(action).lower() else 0.0
    
    def _is_task_complete(self, action: Any) -> bool:
        """Check if task is completed."""
        return "done" in str(action).lower()
    
    @staticmethod
    def from_json(info: Dict) -> "MyCustomEnvironment":
        """Create environment from JSON configuration."""
        return MyCustomEnvironment(
            task=info.get("task", {}),
            max_steps=info.get("max_steps", 10)
        )
    
    @staticmethod
    def is_multithread_safe() -> bool:
        return True  # Specify thread safety
```

## Best Practices

1. **Environment Design**: Keep environments focused on specific task types and evaluation criteria

2. **Thread Safety**: Mark environments as thread-safe only if they truly support parallel execution

3. **Resource Management**: Always implement proper cleanup in the `close()` method

4. **JSON Serialization**: Ensure all environment parameters can be serialized for reproducibility

5. **Error Handling**: Implement robust error handling for external dependencies (Docker, web browsers, etc.)

6. **Testing**: Test environments with various action types and edge cases

7. **Documentation**: Clearly document observation and action spaces, reward structures, and termination conditions

## Next Steps

- Learn about [Rewards](rewards.md) for evaluating agent performance
- Explore the [API Reference](../api/environments.md) for detailed environment documentation
- See [Examples](../examples/environments.md) for complete environment implementations 