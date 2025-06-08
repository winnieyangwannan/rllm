# Environment API Reference

This page provides detailed API documentation for all environments in rLLM.

## Base Classes

### BaseEnv

The abstract base class that all environments inherit from.

```python
from rllm.environments.base.base_env import BaseEnv
```

#### Properties

##### `idx`

Environment index for batch processing.

**Type:** `Any`

**Usage:**
```python
env = MyEnvironment()
env.idx = 5  # Set index for batch tracking
print(env.idx)  # Get current index
```

#### Methods

##### `reset(seed: int = 0, **kwargs) -> Tuple[Any, Dict]`

Reset the environment to an initial state and return observation and info.

**Parameters:**
- `seed` (int): Random seed for environment initialization. Default: 0
- `**kwargs`: Additional arguments specific to the environment

**Returns:** `Tuple[observation, info]` - Initial observation and environment info

##### `step(action: Any) -> Tuple[Any, float, bool, bool, Dict]`

Execute one time step within the environment.

**Parameters:**
- `action` (Any): Action provided by the agent

**Returns:** `Tuple[observation, reward, terminated, truncated, info]`
- `observation` (Any): Environment observation after action
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode ended due to task completion
- `truncated` (bool): Whether episode ended due to time limits
- `info` (Dict): Additional information about the step

##### `close()`

Perform any necessary cleanup of environment resources.

**Returns:** None

##### `from_json(info: Dict) -> BaseEnv` (static)

Create an environment instance from a JSON-like dictionary.

**Parameters:**
- `info` (Dict): Dictionary containing environment configuration

**Returns:** Initialized environment instance

##### `is_multithread_safe() -> bool` (static)

Indicate whether the environment can be used safely in parallel threads.

**Returns:** `bool` - True if thread-safe, False otherwise

---

### MultiTurnEnvironment

Abstract base class for multi-turn interactions.

```python
from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
```

#### Constructor

```python
MultiTurnEnvironment(task: Optional[Dict] = None, max_turns: int = 3, **kwargs)
```

**Parameters:**
- `task` (Optional[Dict]): Task configuration dictionary
- `max_turns` (int): Maximum number of turns before termination. Default: 3
- `**kwargs`: Additional configuration parameters

#### Attributes

- `task` (Dict): Current task configuration
- `max_turns` (int): Maximum allowed turns
- `current_turn` (int): Current turn number
- `done` (bool): Whether the episode is complete
- `history` (List): History of actions taken

#### Abstract Methods

##### `get_reward_and_next_obs(task: Dict, action: Any) -> Tuple[float, Dict]`

Compute reward and generate next observation based on task and action.

**Parameters:**
- `task` (Dict): Task configuration
- `action` (Any): Action taken by agent

**Returns:** `Tuple[reward, next_observation]`

#### Example Implementation

```python
class MyMultiTurnEnv(MultiTurnEnvironment):
    def get_reward_and_next_obs(self, task, action):
        reward = evaluate_action(action, task)
        next_obs = generate_observation(task, action)
        return reward, next_obs
```

---

## Environment Implementations

### SingleTurnEnvironment

Environment for single-interaction tasks like question answering.

```python
from rllm.environments import SingleTurnEnvironment
```

#### Constructor

```python
SingleTurnEnvironment(task: Optional[Dict] = None, reward_fn: Optional[RewardFunction] = None, **kwargs)
```

**Parameters:**
- `task` (Optional[Dict]): Task containing at least a "question" field
- `reward_fn` (Optional[RewardFunction]): Custom reward function. Default: `rllm_reward_fn`
- `**kwargs`: Additional parameters passed to MultiTurnEnvironment

#### Attributes

- `reward_fn` (RewardFunction): Function used to compute rewards

#### Methods

##### `get_reward_and_next_obs(task: Dict, action: Any) -> Tuple[float, Dict]`

Compute reward using the configured reward function.

**Parameters:**
- `task` (Dict): Task information including question and ground truth
- `action` (Any): Agent's response

**Returns:** `Tuple[reward, empty_dict]` - Reward and empty next observation

#### Example Usage

```python
env = SingleTurnEnvironment(
    task={"question": "What is 2+2?", "ground_truth": "4"},
    reward_fn=my_custom_reward_fn
)

obs, info = env.reset()
reward = env.step("The answer is 4")
```

---

### ToolEnvironment

Environment for tool-using agents with function calling support.

```python
from rllm.environments import ToolEnvironment
```

#### Constructor

```python
ToolEnvironment(task: Optional[Dict] = None, tools: List[str] = [], reward_fn: Optional[RewardFunction] = None, max_steps: int = 10)
```

**Parameters:**
- `task` (Optional[Dict]): Task configuration with "question" field
- `tools` (List[str]): List of available tool names
- `reward_fn` (Optional[RewardFunction]): Reward function for final evaluation
- `max_steps` (int): Maximum number of steps before truncation. Default: 10

#### Attributes

- `tools` (MultiTool): Multi-tool interface for executing tools
- `step_count` (int): Current step number
- `max_steps` (int): Maximum allowed steps
- `reward_fn` (RewardFunction): Reward evaluation function

#### Methods

##### `step(action: Union[List[Dict], str, Dict]) -> Tuple[Dict, float, bool, Dict]`

Execute tool calls or handle finish actions.

**Parameters:**
- `action` (Union[List[Dict], str, Dict]): Tool calls or finish response

**Returns:** Tool outputs, reward, done status, and info

**Action Formats:**
- Tool calls: `[{"id": "call_1", "function": {"name": "calculator", "arguments": "{\"expr\": \"2+2\"}"}}]`
- Finish action: `[{"function": {"name": "finish", "arguments": "{\"response\": \"The answer is 4\"}"}}]`
- Direct response: `"The answer is 4"`

#### Example Usage

```python
env = ToolEnvironment(
    task={"question": "What is 15 * 23?"},
    tools=["calculator"],
    max_steps=5
)

obs, info = env.reset()
# obs: {"question": "What is 15 * 23?"}

tool_calls = [{
    "id": "call_1", 
    "function": {"name": "calculator", "arguments": '{"expr": "15*23"}'}
}]
next_obs, reward, done, info = env.step(tool_calls)
# next_obs: {"tool_outputs": {"call_1": "345"}}
```

---

### BrowserGym

Environment for web browsing tasks using the BrowserGym library.

```python
from rllm.environments import BrowserGym
```

#### Constructor

```python
BrowserGym(env_id: str = "browsergym/openended", task: Optional[Dict] = None, **env_kwargs)
```

**Parameters:**
- `env_id` (str): BrowserGym environment identifier. Default: "browsergym/openended"
- `task` (Optional[Dict]): Task configuration for openended environments
- `**env_kwargs`: Additional arguments passed to BrowserGym environment

#### Attributes

- `env` (gym.Env): Underlying BrowserGym environment instance
- `task` (Dict): Task configuration
- `env_kwargs` (Dict): Environment creation parameters

#### Methods

##### `step(action: str) -> Tuple[Any, float, bool, Dict]`

Execute a browser action.

**Parameters:**
- `action` (str): BrowserGym action string (e.g., "click [123]", "type [456]; Hello")

**Returns:** Observation, reward, done status, and info from BrowserGym

#### Thread Safety

**Not multithread safe** due to browser instance limitations.

#### Example Usage

```python
env = BrowserGym(
    env_id="browsergym/openended",
    task={"instruction": "Search for information about Python"}
)

obs, info = env.reset()
next_obs, reward, done, info = env.step("click [search_button]")
```

---

### CompetitionCodingEnv

Environment for competitive programming tasks with iterative code improvement.

```python
from rllm.environments import CompetitionCodingEnv
```

#### Constructor

```python
CompetitionCodingEnv(task: Optional[Dict] = None, max_turns: int = 2, reward_bonus_coeff: float = 0.0, **kwargs)
```

**Parameters:**
- `task` (Optional[Dict]): Programming task with question and ground truth
- `max_turns` (int): Maximum coding attempts. Default: 2
- `reward_bonus_coeff` (float): Coefficient for reward shaping. Default: 0.0
- `**kwargs`: Additional parameters

#### Attributes

- `reward_fn` (RewardFunction): Code evaluation function
- `prev_reward` (float): Previous attempt's reward for shaping
- `reward_bonus_coeff` (float): Reward improvement bonus coefficient

#### Methods

##### `get_reward_and_next_obs(task: Dict, action: str) -> Tuple[float, Dict]`

Evaluate code solution against test cases.

**Parameters:**
- `task` (Dict): Programming task configuration
- `action` (str): Code solution from agent

**Returns:** `Tuple[reward, metadata]` - Test evaluation results

#### Reward Shaping

The environment applies reward shaping based on improvement:
```
final_reward = current_reward + bonus_coeff * (current_reward - previous_reward)
```

#### Example Usage

```python
env = CompetitionCodingEnv(
    task={
        "question": "Write a function to reverse a string",
        "ground_truth": test_cases_data
    },
    max_turns=3,
    reward_bonus_coeff=0.5
)

obs, info = env.reset()
code = "def reverse_string(s): return s[::-1]"
next_obs, reward, done, info = env.step(code)
```

---

### SWEEnv

Software engineering environment for repository-based coding tasks.

```python
from rllm.environments import SWEEnv
```

#### Constructor

```python
SWEEnv(entry: Optional[Dict] = None, dataset: Optional[Dataset] = None, idx: Optional[int] = None, step_timeout: int = 90, reward_timeout: int = 300, backend: str = "kubernetes", delete_image: bool = False, verbose: bool = False)
```

**Parameters:**
- `entry` (Optional[Dict]): Specific dataset entry to use
- `dataset` (Optional[Dataset]): Dataset containing tasks (uses default if None)
- `idx` (Optional[int]): Index of task in dataset (random if None)
- `step_timeout` (int): Timeout for each step in seconds. Default: 90
- `reward_timeout` (int): Timeout for reward computation. Default: 300
- `backend` (str): Execution backend ("kubernetes" or "docker"). Default: "kubernetes"
- `delete_image` (bool): Whether to delete Docker image on close. Default: False
- `verbose` (bool): Enable verbose logging. Default: False

#### Attributes

- `env` (RepoEnv): R2E-Gym repository environment
- `entry` (Dict): Current task entry
- `total_steps` (int): Total steps taken in episode

#### Methods

##### `step(action: Union[str, Action]) -> Tuple[str, float, bool, Dict]`

Execute an action in the repository environment.

**Parameters:**
- `action` (Union[str, Action]): XML-formatted action string or Action object

**Returns:** Observation text, reward, done status, and info

##### `compute_final_reward() -> float`

Compute final reward by evaluating the repository changes.

**Returns:** Final evaluation score

#### Available Actions

- `file_editor`: View, create, edit files
- `search`: Search for terms in files
- `execute_bash`: Run bash commands
- `finish`: Complete the task

#### Example Usage

```python
env = SWEEnv(
    entry={"problem_statement": "Fix the bug in utils.py"},
    backend="kubernetes",
    verbose=True
)

obs, info = env.reset()
action = "<function=file_editor><parameter=command>view</parameter><parameter=path>/repo/utils.py</parameter></function>"
next_obs, reward, done, info = env.step(action)
```

---

### FrozenLakeEnv

Grid-world reinforcement learning environment with text observations.

```python
from rllm.environments import FrozenLakeEnv
```

#### Constructor

```python
FrozenLakeEnv(size: int = 8, p: float = 0.8, is_slippery: bool = False, max_steps: int = 5, seed: int = 42, desc: Optional[List[str]] = None, **kwargs)
```

**Parameters:**
- `size` (int): Grid size (size × size). Default: 8
- `p` (float): Probability of frozen tiles vs holes. Default: 0.8
- `is_slippery` (bool): Whether movement is stochastic. Default: False
- `max_steps` (int): Maximum steps before truncation. Default: 5
- `seed` (int): Random seed for map generation. Default: 42
- `desc` (Optional[List[str]]): Custom map description. Default: None (random)
- `**kwargs`: Additional Gymnasium parameters

#### Attributes

- `goal_position` (Tuple[int, int]): Location of goal tile
- `ACTION_SPACE` (gym.Space): Discrete action space {1, 2, 3, 4}
- `MAP_LOOKUP` (Dict): Mapping from tiles to integers
- `GRID_LOOKUP` (Dict): Mapping for text rendering
- `ACTION_LOOKUP` (Dict): Action names

#### Action Space

- `1`: Left
- `2`: Down  
- `3`: Right
- `4`: Up

#### Observation Space

Text-based grid representation showing player (P), frozen tiles (_), holes (O), and goal (G).

#### Methods

##### `step(action: str) -> Tuple[str, float, bool, bool, Dict]`

Execute movement action.

**Parameters:**
- `action` (str): Movement direction ("Left", "Down", "Right", "Up")

**Returns:** Grid text, reward, terminated, truncated, info

##### `render(mode: str = 'tiny_rgb_array') -> str`

Render the current environment state.

**Parameters:**
- `mode` (str): Render mode. Default: "tiny_rgb_array"

**Returns:** Text representation of the grid

#### Example Usage

```python
env = FrozenLakeEnv(
    size=4,
    p=0.9,
    is_slippery=False,
    max_steps=10
)

obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step("Right")
print(env.render())
```

---

## Environment Utilities

### Trajectory Processing

Utilities for computing trajectory statistics and returns.

#### `compute_trajectory_reward(trajectory: Trajectory) -> Trajectory`

Compute total reward for a trajectory.

**Parameters:**
- `trajectory` (Trajectory): Agent trajectory object

**Returns:** Updated trajectory with total reward

#### `compute_mc_return(trajectory: Trajectory, gamma: float = 0.95) -> Trajectory`

Compute Monte Carlo returns for each step in a trajectory.

**Parameters:**
- `trajectory` (Trajectory): Agent trajectory object
- `gamma` (float): Discount factor. Default: 0.95

**Returns:** Updated trajectory with MC returns

#### Example Usage

```python
from rllm.environments.env_utils import compute_trajectory_reward, compute_mc_return

# Process trajectory
trajectory = compute_trajectory_reward(agent.trajectory)
trajectory = compute_mc_return(trajectory, gamma=0.99)

print(f"Total reward: {trajectory.reward}")
for step in trajectory.steps:
    print(f"Step {step.step}: MC return = {step.mc_return}")
```

### Parallel Processing

#### `parallel_task_manager(func: Callable, items: List[Any], max_workers: int = 32)`

Context manager for parallel execution of tasks.

**Parameters:**
- `func` (Callable): Function to execute on each item
- `items` (List[Any]): Items to process
- `max_workers` (int): Maximum thread pool workers. Default: 32

**Yields:** `List[Tuple[int, Any]]` - List of (index, result) tuples

#### Example Usage

```python
from rllm.environments.env_utils import parallel_task_manager

def process_task(task_config):
    env = MyEnvironment(**task_config)
    return env.run_episode()

tasks = [{"seed": i} for i in range(100)]

with parallel_task_manager(process_task, tasks, max_workers=16) as results:
    for idx, result in results:
        print(f"Task {idx}: {result}")
```

---

## Configuration and Serialization

### JSON Configuration

All environments support creation from JSON configurations:

```python
# Environment configuration
config = {
    "task": {"question": "What is AI?"},
    "max_steps": 5,
    "reward_fn": "custom_reward"
}

# Create from config
env = MyEnvironment.from_json(config)

# Serialize config
import json
with open("env_config.json", "w") as f:
    json.dump(config, f)
```

### Batch Environment Management

Environments support batch indexing for parallel processing:

```python
# Create batch of environments
envs = []
for i, task in enumerate(tasks):
    env = SingleTurnEnvironment(task=task)
    env.idx = i  # Set batch index
    envs.append(env)

# Parallel processing
def run_env(env):
    obs, info = env.reset()
    action = generate_action(obs)
    return env.idx, env.step(action)

import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(run_env, envs))
```
