# Agent Execution Engine

The Agent Execution Engine is the central orchestrator in rLLM that manages interactions between agents and environments. It handles batched execution, trajectory collection, and provides both synchronous and asynchronous execution modes for scalable RL training.

## Overview

The Agent Execution Engine serves as the runtime coordinator that:

- **Orchestrates interactions** between multiple agent-environment pairs
- **Manages trajectories** by collecting step-by-step interaction data
- **Handles parallelization** for efficient batch processing
- **Supports multiple output formats** for different use cases
- **Provides retry mechanisms** for robust execution

## Architecture

The execution engine consists of two main implementations:

### AgentExecutionEngine (Synchronous)
The standard synchronous engine that processes agent-environment interactions in batch mode:

```python
from rllm.engine.agent_execution_engine import AgentExecutionEngine

engine = AgentExecutionEngine(
    rollout_engine=rollout_engine,
    engine_name="verl",
    tokenizer=tokenizer,
    config=config,
    agents=agents,
    envs=environments,
    max_steps=10,
    max_response_length=2048,
    gamma=0.95
)

# Generate trajectories
trajectories = engine.generate_trajectories(mode="Text")
```

### AsyncAgentExecutionEngine (Asynchronous)
An asynchronous version designed for high-throughput scenarios:

```python
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

async_engine = AsyncAgentExecutionEngine(
    rollout_engine=rollout_engine,
    engine_name="verl",
    tokenizer=tokenizer,
    config=config,
    agents=agents,
    envs=environments,
    max_workers=64
)

# Generate trajectories asynchronously
async def run_trajectories():
    trajectories = []
    async for trajectory in async_engine.trajectory_generator(mode="Text"):
        trajectories.append(trajectory)
    return trajectories
```

## Key Features

### Batched Execution
The engine efficiently processes multiple agent-environment pairs in parallel:

- **Parallel environments**: Multiple environments run simultaneously
- **Batched model inference**: Groups model requests for efficiency
- **Resource management**: Controls thread pools and worker limits

### Trajectory Management
Comprehensive trajectory tracking and processing:

- **Step-by-step recording**: Captures observations, actions, rewards, and metadata
- **Multiple return formats**: Text, Token, Conversation, and Step modes
- **Reward computation**: Automatic trajectory reward and Monte Carlo return calculation

### Model Integration
Seamless integration with different model backends:

- **veRL support**: Native integration with veRL distributed inference
- **OpenAI API support**: Compatible with OpenAI-style APIs
- **Flexible tokenization**: Handles various tokenizers and chat templates

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | int | 5 | Maximum number of steps per trajectory |
| `max_response_length` | int | 16384 | Maximum tokens in model responses |
| `max_prompt_length` | int | 2048 | Maximum tokens in input prompts |
| `gamma` | float | 0.95 | Discount factor for Monte Carlo returns |
| `n_parallel_agents` | int | None | Number of parallel agent-environment pairs |

### Execution Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retry_limit` | int | 1 | Number of retry attempts for failed trajectories |
| `api_retries` | int | 3 | Number of API retry attempts |
| `trajectory_timeout` | int | 1e9 | Timeout for trajectory execution (seconds) |
| `max_workers` | int | 16 | Maximum number of worker threads |

### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enforce_max_prompt_length` | bool | False | Apply prompt length limit per step |
| `sampling_params` | dict | None | Model sampling parameters |
| `disable_thinking` | bool | False | Disable thinking tokens in chat templates |

## Output Formats

The execution engine supports multiple output formats to suit different use cases:

### Text Mode
Returns structured trajectory objects with full interaction history:

```python
trajectories = engine.generate_trajectories(mode="Text")
for trajectory in trajectories:
    for step in trajectory.steps:
        print(f"Observation: {step.observation}")
        print(f"Action: {step.action}")
        print(f"Reward: {step.reward}")
```

### Token Mode
Returns tokenized data optimized for model training:

```python
token_data = engine.generate_trajectories(mode="Token")
for item in token_data:
    prompt_tokens = item["prompt_tokens"]  # torch.Tensor
    response_tokens = item["response_tokens"]  # torch.Tensor
    response_masks = item["response_masks"]  # torch.Tensor
    training_reward = item["training_reward"]  # float
```

### Conversation Mode
Returns raw conversation messages in ChatML format:

```python
conversations = engine.generate_trajectories(mode="Conversation")
for messages in conversations:
    for message in messages:
        print(f"{message['role']}: {message['content']}")
```

### Step Mode
Returns step-by-step prompt-response pairs:

```python
step_data = engine.generate_trajectories(mode="Step")
for episode in step_data:
    for step in episode["steps"]:
        print(f"Prompt: {step['prompt']}")
        print(f"Response: {step['response']}")
```

## Usage Examples

### Basic Synchronous Usage

```python
from rllm.engine.agent_execution_engine import AgentExecutionEngine

# Initialize engine
engine = AgentExecutionEngine(
    rollout_engine=rollout_engine,
    engine_name="verl",
    tokenizer=tokenizer,
    config=config,
    agents=agents,
    envs=environments,
    max_steps=10,
    gamma=0.95
)

# Generate trajectories
trajectories = engine.generate_trajectories(
    reset_seed=42,
    mode="Text"
)

# Process results
for i, trajectory in enumerate(trajectories):
    print(f"Trajectory {i}: Reward = {trajectory.reward}")
```

### Asynchronous High-Throughput Usage

```python
import asyncio
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

async def main():
    # Initialize async engine
    engine = AsyncAgentExecutionEngine(
        rollout_engine=rollout_engine,
        engine_name="verl",
        tokenizer=tokenizer,
        config=config,
        agents=agents,
        envs=environments,
        max_workers=64
    )
    
    # Generate trajectories asynchronously
    trajectories = []
    async for trajectory in engine.trajectory_generator(mode="Token"):
        trajectories.append(trajectory)
        print(f"Collected {len(trajectories)} trajectories")
    
    return trajectories

# Run async execution
trajectories = asyncio.run(main())
```

### Training Data Collection

```python
# Collect training data in token format
training_data = engine.generate_trajectories(
    mode="Token",
    reset_seed=0
)

# Prepare for training
prompt_tokens = [item["prompt_tokens"] for item in training_data]
response_tokens = [item["response_tokens"] for item in training_data]
rewards = [item["training_reward"] for item in training_data]

# Use with your training loop
trainer.train(prompt_tokens, response_tokens, rewards)
```

## Best Practices

### Performance Optimization

1. **Choose appropriate batch sizes**: Balance memory usage with throughput
2. **Use async engine for high concurrency**: Better for I/O-bound workloads
3. **Configure worker limits**: Match your hardware capabilities
4. **Monitor resource usage**: Watch memory and CPU utilization

### Error Handling

1. **Set appropriate retry limits**: Handle transient failures gracefully
2. **Configure timeouts**: Prevent hanging trajectories
3. **Monitor trajectory completion**: Track success/failure rates
4. **Implement fallback strategies**: Handle edge cases in environments

### Memory Management

1. **Control trajectory length**: Use `max_steps` and token limits
2. **Choose appropriate output modes**: Token mode is more memory-efficient
3. **Process trajectories in batches**: Avoid accumulating too much data
4. **Clean up resources**: Ensure proper environment cleanup

### Scalability Considerations

1. **Distribute across machines**: Use veRL for multi-node scaling
2. **Balance load**: Ensure even distribution of computational work
3. **Monitor bottlenecks**: Identify and address performance limitations
4. **Optimize model serving**: Use efficient inference backends

## Integration with Training

The execution engine integrates seamlessly with rLLM's training pipeline:

```python
# Collect trajectories
trajectories = engine.generate_trajectories(mode="Token")

# Use with PPO trainer
from rllm.trainer.ppo_trainer import PPOTrainer

trainer = PPOTrainer(
    config=config,
    tokenizer=tokenizer,
    policy_model=policy_model,
    value_model=value_model
)

# Train on collected trajectories
trainer.train_step(trajectories)
```

## Troubleshooting

### Common Issues

**Trajectory Timeouts**: Increase `trajectory_timeout` or optimize environment performance
**Memory Errors**: Reduce batch size or use Token mode for efficiency  
**API Rate Limits**: Increase `api_retries` or implement backoff strategies
**Environment Errors**: Ensure environments are thread-safe for parallel execution

### Debugging Tips

1. **Enable verbose logging**: Monitor trajectory execution progress
2. **Use smaller batch sizes**: Isolate issues with individual trajectories
3. **Check environment compatibility**: Verify thread safety for async execution
4. **Monitor resource usage**: Watch for memory leaks or resource exhaustion

## Next Steps

- Learn about [Agents](agents.md) that the engine orchestrates
- Explore [Environments](environments.md) that provide the interaction context
- See [Training](training.md) for integration with RL algorithms
- Check the [API Reference](../api/execution-engine.md) for detailed parameter documentation 