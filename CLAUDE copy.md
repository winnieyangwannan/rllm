# rLLM Architecture Overview
2025.10
rLLM is a reinforcement learning framework for post-training language agents. It provides a modular, composable architecture for building agentic systems that can learn through interaction with environments and model-generated trajectories.

## High-Level Architecture

rLLM's core design follows a **modular composition pattern** where agents, environments, and workflows orchestrate training loops. The system has two primary execution modes:

1. **Simple Mode**: Direct agent-environment interaction via `AgentExecutionEngine`
2. **Workflow Mode**: Complex multi-agent, multi-environment workflows orchestrated by `AgentWorkflowEngine`

```
┌─────────────────────────────────────────────────────────────────┐
│                     rLLM Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  AgentTrainer                                                     │
│  ├── Config Management (via Hydra)                              │
│  ├── Dataset Loading                                            │
│  └── Training Orchestration (via Ray)                           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Two Execution Paths                              │   │
│  │                                                          │   │
│  │  Path 1: AgentExecutionEngine (Simple)                  │   │
│  │  ├─ Direct agent-env interaction                        │   │
│  │  └─ Minimal orchestration                               │   │
│  │                                                          │   │
│  │  Path 2: AgentWorkflowEngine + Workflow (Complex)       │   │
│  │  ├─ Workflow pool management                            │   │
│  │  ├─ Multi-agent/environment composition                 │   │
│  │  ├─ Parallel task execution                             │   │
│  │  └─ Custom execution logic per workflow                 │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Shared Components                                │   │
│  │                                                          │   │
│  │  RolloutEngine (OpenAI/Verl)                            │   │
│  │  └─ Async model inference                               │   │
│  │                                                          │   │
│  │  Step/Trajectory/Episode Data Structures                │   │
│  │  └─ Trajectory collection & serialization               │   │
│  │                                                          │   │
│  │  Reward Functions                                        │   │
│  │  └─ Math, Code, Search, Custom evaluations              │   │
│  │                                                          │   │
│  │  Tools Integration                                       │   │
│  │  └─ Multi-tool orchestration & execution                │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Verl Integration (Training Backend)                            │
│  └─ PPO training with distributed compute                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Structures: Step, Trajectory, Episode

These form the foundation of trajectory collection and serve as the unified interface between agents, environments, and training:

**Step** (`rllm/agents/agent.py`):
- Represents a single agent decision point
- Contains:
  - `chat_completions`: Message history sent to model
  - `observation`: Current world state
  - `thought`: Agent's reasoning (if enabled)
  - `action`: Agent's action (tool calls, text, etc.)
  - `model_response`: Raw model output
  - `model_output`: Structured output with token info
  - `reward`: Step-level reward (filled by workflow)
  - `done`: Termination flag
  - `mc_return`: Monte Carlo discounted return
- Serializable via `to_dict()` / `from_dict()` for training data

**Trajectory**:
- Sequence of Steps representing a complete agent run
- Aggregates steps into an ordered history
- Includes task metadata and trajectory-level reward
- Can detect if trajectory is "cumulative" (chat_completions prefix)

**Episode**:
- Container for multiple Trajectories (from multiple agents in workflow)
- Stores:
  - Task specification and result
  - Correctness flag
  - Termination reason
  - Metrics (e.g., per-agent accuracy)
  - Error details
- Represents one rollout/evaluation of a workflow

### 2. Agents: BaseAgent

The `BaseAgent` abstract class defines the agent interface:

```python
class BaseAgent:
    # State Properties
    chat_completions: list[dict] -> message history for model
    trajectory: Trajectory -> agent's recorded history
    
    # State Mutations
    update_from_env(observation, reward, done, info)
    update_from_model(response: str) -> Action
    reset()
    
    # State Access
    get_current_state() -> Step | None
```

**Key Abstraction**: Agents maintain internal state and expose it through:
- `chat_completions`: What gets sent to the model
- `trajectory`: What gets recorded for training

**Common Implementations**:
- **MathAgent**: State machine tracking question->response flow, supports thinking tokens
- **ToolAgent**: Manages tool call parsing, multi-turn tool use, system prompts
- **CodeAgent**: Specialized for code generation tasks
- **WebarenaAgent**: Browser automation with DOM observations

**Agent Design Pattern**:
```python
# 1. Agent initialized with system prompt, tools, etc.
agent = MathAgent()

# 2. Reset for new episode
agent.reset()

# 3. Receive initial observation
agent.update_from_env(observation, reward=0, done=False, info={})

# 4. Model generates response
response = model(agent.chat_completions)

# 5. Agent parses and updates state
action = agent.update_from_model(response)

# 6. Environment executes action
obs, reward, done, info = env.step(action)

# 7. Repeat steps 3-6 until done
```

### 3. Environments: BaseEnv

The `BaseEnv` abstract class provides the standard Gym interface:

```python
class BaseEnv:
    reset(task: dict) -> (observation, info)
    step(action) -> (observation, reward, done, info)
    from_dict(config: dict) -> BaseEnv  # Factory method
    is_multithread_safe() -> bool
```

**Key Concepts**:
- Standard Gym interface but with task-aware initialization
- `reset(task)` receives task dict (question, context, etc.)
- `is_multithread_safe()` indicates if env can be shared across threads
- Concrete implementations add reward functions

**Common Implementations**:
- **SingleTurnEnvironment**: One-shot evaluation with reward function
- **MultiTurnEnvironment**: Multi-step interaction until termination
- **ToolEnvironment**: Executes tool calls and returns outputs
- **FrozenLakeEnv**: Simple grid world (testing)

**Reward Integration**:
```python
class SingleTurnEnvironment(MultiTurnEnvironment):
    def __init__(self, reward_fn: RewardFunction):
        self.reward_fn = reward_fn
    
    def step(self, action):
        reward_output = self.reward_fn(task_info=self.task, action=action)
        return observation, reward_output.reward, done, info
```

### 4. Workflows: Orchestrating Agents and Environments

**Workflow** (`rllm/workflows/workflow.py`) is the orchestration layer that:
- Manages composition of agents and environments
- Handles async/await patterns
- Collects trajectories
- Computes rewards
- Tracks termination reasons
- Post-processes episodes

**Base Workflow Pattern**:
```python
class Workflow(ABC):
    def __init__(self, rollout_engine: RolloutEngine, executor: ThreadPoolExecutor, ...):
        self.rollout_engine = rollout_engine
        self.executor = executor
        
    async def run(self, task: dict, uid: str) -> Episode | None:
        # Implement custom execution logic
        pass
    
    async def run_with_termination_handling(self, task, uid):
        # Wraps run() with error handling and post-processing
        pass
```

**Workflow Lifecycle**:
1. `reset(task, uid)` - Initialize agents/envs for new task
2. `run(task, uid)` - Execute custom agent-environment interaction
3. `commit(agent/trajectory)` - Record trajectories
4. `collect_trajectories()` - Gather all recorded trajectories
5. `postprocess_episode()` - Compute rewards, MC returns, metrics

**Built-in Workflow Implementations**:

**SingleTurnWorkflow**:
```
reset() -> env.reset(task)
observe initial state
model_response = rollout_engine.get_model_response(messages)
action = agent.update_from_model(response)
obs, reward, done, info = env.step(action)
agent.update_from_env(obs, reward, done, info)
```

**MultiTurnWorkflow**:
```
reset() -> env.reset(task)
observe initial state
LOOP (max_steps times):
    model_response = rollout_engine.get_model_response(messages)
    action = agent.update_from_model(response)
    obs, reward, done, info = env.step(action)
    agent.update_from_env(obs, reward, done, info)
    if done: break
```

**Custom Workflow Example** (StrandsWorkflow):
```python
class StrandsWorkflow(Workflow):
    async def run(self, task: dict, uid: str, **kwargs):
        self.reset(task=task, uid=uid)
        result = self.agent(task_text)  # Direct agent call
        self.commit(agent=self.agent)
        raise TerminationEvent(TerminationReason.ENV_DONE)
```

**Reward Computation in Workflows**:
```python
# Step-level rewards come from environment
# Workflow computes trajectory-level aggregation
def compute_trajectory_reward(trajectory):
    trajectory.reward = sum(step.reward for step in trajectory.steps)

# Reward shaping and discounting (optional)
def adjust_step_rewards(trajectory):
    if reward_bonus_coeff > 0:
        # Reward bonus on reward differences
        for i in range(1, len(steps)):
            steps[i].reward += bonus * (steps[i].reward - steps[i-1].reward)
    
    if gamma > 0:
        # Monte Carlo returns (discounted)
        G = 0
        for step in reversed(trajectory.steps):
            G = step.reward + gamma * G
            step.mc_return = G
```

### 5. Rollout Engines: Async Model Inference

**RolloutEngine** is the abstraction for model inference:

```python
class RolloutEngine:
    async def get_model_response(messages: list[dict]) -> ModelOutput
```

**ModelOutput** structure:
```python
@dataclass
class ModelOutput:
    text: str                    # Full text response
    content: str                 # Content without reasoning
    reasoning: str               # Chain-of-thought/thinking content
    tool_calls: list[ToolCall]  # Parsed function calls
    prompt_ids: list[int]        # Tokenized prompt
    completion_ids: list[int]    # Tokenized completion
    prompt_length: int
    completion_length: int
    finish_reason: str           # "stop", "length", "tool_calls"
```

**Implementations**:
- **OpenAIEngine**: Calls OpenAI API compatible servers (vLLM, SGLang, etc.)
- **VerlEngine**: Uses Verl's distributed inference for training
- **FireworksEngine**: Calls Fireworks API

**Key Feature**: Returns structured output with token counts, enabling length constraints and token accounting.

### 6. Agents Execution Engine (Simple Path)

For simpler scenarios without custom workflows, `AgentExecutionEngine` directly manages agent-environment pairs:

```python
class AgentExecutionEngine:
    def __init__(
        self,
        agent_class: type,
        agent_args: dict,
        env_class: type,
        env_args: dict,
        rollout_engine: RolloutEngine,
        n_parallel_agents: int = 1,
        max_steps: int = 5,
        gamma: float = 0.2,  # discount factor
    ):
        self.agents = [agent_class(**agent_args) for _ in range(n_parallel_agents)]
        self.envs = [env_class(**env_args) for _ in range(n_parallel_agents)]
        self.rollout_engine = rollout_engine
```

**Execution Loop**:
- Maintains parallel agent-env pairs
- Synchronously steps through interactions
- Computes MC returns with gamma
- Returns trajectories for training

### 7. Workflow Engine: Parallel Orchestration

For complex agentic programs, `AgentWorkflowEngine` manages parallel workflows:

```python
class AgentWorkflowEngine:
    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine: RolloutEngine,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
    ):
        self.workflow_queue = asyncio.Queue(maxsize=n_parallel_tasks)
        # Pool of workflow instances ready for tasks
```

**Key Features**:
1. **Workflow Pool**: Maintains N pre-instantiated workflow instances
2. **Task Queue**: Processes tasks asynchronously
3. **Retry Logic**: Retries on ERROR termination reasons (not on env-induced terminations)
4. **Thread Safety**: Validates all envs are multithread-safe before parallel execution
5. **Parallel Processing**: Concurrent execution of multiple workflows

**Task Processing with Retry**:
```python
async def process_task_with_retry(task, task_id, rollout_idx):
    for retry_attempt in range(1, retry_limit + 1):
        workflow = await queue.get()
        try:
            episode = await workflow.run_with_termination_handling(task, uid)
            if episode.termination_reason != TerminationReason.ERROR:
                return episode  # Success or env termination
            if retry_attempt < retry_limit:
                continue  # Retry
        finally:
            queue.put_nowait(workflow)
```

### 8. Reward Functions

Rewards evaluate whether the agent's action solves the task:

**RewardFunction Protocol**:
```python
def reward_fn(task_info: dict, action: str) -> RewardOutput
```

**RewardOutput**:
```python
@dataclass
class RewardOutput:
    reward: float                    # -1.0 to 1.0 typically
    metadata: dict = {}              # Custom eval info
    is_correct: bool | None = None   # Correctness flag
```

**Built-in Reward Functions**:
- **math_reward_fn**: Evaluates math solutions (AIME, Hendrycks, etc.)
- **code_reward_fn**: Executes code and checks outputs
- **search_reward_fn**: Evaluates information retrieval
- **zero_reward**: Default no-op (all rewards 0.0)

**Custom Rewards Example**:
```python
def custom_reward_fn(task_info, action):
    answer = extract_answer(action)
    is_correct = answer == task_info["expected"]
    return RewardOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        metadata={"answer": answer}
    )
```

### 9. Tools Integration

Tools allow agents to call external functions (Python, calculators, web search, etc.):

**Tool Base Class** (`rllm/tools/tool_base.py`):
```python
class Tool:
    def __init__(self, name, description, function):
        self.name = name
        self.function = function
    
    def forward(self, **kwargs) -> ToolOutput:
        # Synchronous execution
        pass
    
    async def async_forward(self, **kwargs) -> ToolOutput:
        # Async execution
        pass
```

**ToolCall**: Parsed function call from model response
```python
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
```

**MultiTool Orchestration** (`rllm/tools/multi_tool.py`):
```python
class MultiTool:
    def __init__(self, tools: list[str] | tool_map: dict):
        # Either load from registry by name or use custom map
        self.tools = {name: tool_class(...) for name, tool_class in tool_map.items()}
    
    def __call__(self, tool_name: str, **kwargs) -> ToolOutput:
        return self.tools[tool_name].forward(**kwargs)
```

**Tool-Aware Environments**:
```python
class ToolEnvironment(BaseEnv):
    def __init__(self, tools: list[str], reward_fn):
        self.tools = MultiTool(tools=tools)
    
    def step(self, action: list[dict]):  # action = tool_calls
        for tool_call in action:
            if tool_call["function"]["name"] == "finish":
                # Evaluation step
                llm_response = tool_call["function"]["arguments"]["response"]
                reward = self.reward_fn(task, llm_response)
                return {}, reward, True, info
        
        # Execute tool calls
        tool_outputs = self._execute_tool_calls(action)
        return {"tool_outputs": tool_outputs}, 0, False, info
```

## Training Orchestration

### AgentTrainer: Entry Point

`AgentTrainer` is the high-level API that users interact with:

```python
trainer = AgentTrainer(
    # Agent/Env Path (simple)
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    agent_args={},
    env_args={"reward_fn": math_reward_fn},
    
    # OR Workflow Path (complex)
    workflow_class=SingleTurnWorkflow,
    workflow_args={
        "agent_cls": MathAgent,
        "env_cls": SingleTurnEnvironment,
    },
    
    config=config,  # Hydra config with training params
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

trainer.train()  # Launches distributed training via Ray
```

**Two Modes**:
1. **Agent/Env Mode**: Direct instantiation, good for simple single-agent scenarios
2. **Workflow Mode**: Custom composition, good for multi-agent or complex logic

### Training Backend: Verl + Ray

rLLM integrates with **Verl** (a distributed RL training framework) and **Ray** for distributed training:

```
AgentTrainer.train()
├── Ray initialization (distributed compute)
├── TaskRunner (Ray actor) processes batches
├── Per-batch:
│   ├── Rollout collection:
│   │   ├── Instantiate N workflows/engines
│   │   ├── Execute N tasks in parallel
│   │   ├── Collect trajectories (Step/Trajectory/Episode)
│   │   └── Convert to Verl DataProto format
│   │
│   └── Training step:
│       ├── Verl PPO trainer processes trajectories
│       ├── Computes gradients with policy loss
│       ├── Updates model weights
│       └── Return updated model
└── Repeat until convergence
```

**Config-Driven**: Uses Hydra configs for:
- Rollout parameters (max_steps, timeout, max_parallel_tasks)
- Training parameters (learning rate, PPO epochs, batch size)
- Data parameters (dataset paths, batch sizes)

## Design Patterns and Conventions

### 1. State Management Pattern
Agents maintain state through composition:
```python
class MathAgent(BaseAgent):
    def __init__(self):
        self._trajectory = Trajectory()  # Immutable record
        self.messages = []               # Mutable working state
    
    @property
    def trajectory(self):
        return self._trajectory  # Expose current trajectory
    
    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
```

### 2. Async/Await Execution
Workflows and engines use async for parallelism:
```python
async def run(self, task, uid):
    obs, info = await self.run_in_executor(self.env.reset, task=task)
    output = await self.rollout_engine.get_model_response(messages)
    obs, reward, done, info = await self.run_in_executor(self.env.step, action)
```

### 3. Factory Methods
Environments implement `from_dict()` for instantiation from config:
```python
@staticmethod
def from_dict(env_args: dict) -> "SingleTurnEnvironment":
    reward_fn = env_args.pop("reward_fn")
    task = env_args
    return SingleTurnEnvironment(task=task, reward_fn=reward_fn)
```

### 4. Termination Events
Workflows signal completion via exceptions:
```python
if output.finish_reason == "length":
    raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

if done:
    raise TerminationEvent(TerminationReason.ENV_DONE)
```

### 5. Trajectory Serialization
All structures support `to_dict()` / `from_dict()` for training data:
```python
# Save trajectory to disk
trajectory_dict = trajectory.to_dict()
json.dump(trajectory_dict, f)

# Load for post-processing
trajectory = Trajectory.from_dict(trajectory_dict)
```

### 6. Task Metadata Propagation
Tasks flow through the entire system:
```
Dataset -> Rollout (task) -> Workflow (task) -> Agent.reset() 
    -> Env.reset(task) -> Reward computation
```

## Interaction Flow Examples

### Simple Example: Math Agent with Single-Turn

```python
# 1. Create trainer
trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    agent_args={},
    env_args={"reward_fn": math_reward_fn},
    config=config,
)
trainer.train()

# What happens internally:
# - AgentExecutionEngine creates parallel (MathAgent, SingleTurnEnv) pairs
# - For each task in dataset:
#   a) env.reset(task) -> question
#   b) agent.update_from_env(observation)
#   c) model(agent.chat_completions) -> response
#   d) agent.update_from_model(response) -> Action
#   e) env.step(action) -> reward, done
#   f) agent.update_from_env({}, reward, done, info)
#   g) Collect trajectory
# - Convert trajectories to training format
# - PPO training step
```

### Complex Example: Tool-Using Agent with Custom Workflow

```python
# 1. Define custom workflow
class MyWorkflow(Workflow):
    async def run(self, task, uid, **kwargs):
        obs, info = await self.run_in_executor(self.env.reset, task=task)
        self.agent.update_from_env(obs, 0, False, info)
        
        for step in range(self.max_steps):
            output = await self.rollout_engine.get_model_response(
                self.agent.chat_completions
            )
            action = self.agent.update_from_model(output.text)
            
            obs, reward, done, info = await self.run_in_executor(
                self.env.step, action
            )
            self.agent.update_from_env(obs, reward, done, info)
            
            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)
        
        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

# 2. Create trainer
trainer = AgentTrainer(
    workflow_class=MyWorkflow,
    workflow_args={
        "agent_cls": ToolAgent,
        "env_cls": ToolEnvironment,
        "agent_args": {"tools": ["python"]},
        "env_args": {"tools": ["python"], "reward_fn": code_reward_fn},
    },
    config=config,
)
trainer.train()

# What happens internally:
# - AgentWorkflowEngine creates pool of MyWorkflow instances
# - Parallel processing of tasks with retry on ERROR
# - Each workflow instance manages agent-env pair
# - Trajectories collected with tool execution logs
```

## Key Extension Points

Developers can customize:

1. **Agent**: Override `update_from_env`, `update_from_model`, `reset`, `chat_completions`, `trajectory`
2. **Environment**: Implement `reset`, `step`, `from_dict`, inherit from `BaseEnv`
3. **Reward Function**: Implement callable matching `RewardFunction` protocol
4. **Workflow**: Inherit from `Workflow`, override `run()` and optionally `reset()`
5. **Tools**: Inherit from `Tool`, implement `forward()` or `async_forward()`
6. **Rollout Engine**: Inherit from `RolloutEngine`, implement `get_model_response()`

## Dataset Integration

`Dataset` provides simple data abstraction:
```python
dataset = Dataset(data=[{"question": "...", "answer": "..."}, ...])
# Can be repeated for evaluation
repeated = dataset.repeat(n=8)
# Automatically converts to Verl format for training
```

## Common Debugging Points

1. **Trajectory is empty**: Check if `agent.reset()` is called before episode
2. **Wrong reward format**: Ensure reward functions return `RewardOutput` not float
3. **Termination issues**: Check termination reasons in episode post-processing
4. **Multithread errors**: Verify `env.is_multithread_safe()` returns True for parallel workflows
5. **Missing tool outputs**: Ensure tool_calls are properly formatted with "function" key

## Dependencies and Integration

- **Verl**: Distributed training backend (PPO, DPO, etc.)
- **Ray**: Parallel task execution and distributed compute
- **Hydra**: Configuration management
- **Transformers**: Tokenizers and model handling
- **Pydantic**: Validation (in some components)
- **OpenAI API**: Optional, for API-based inference

