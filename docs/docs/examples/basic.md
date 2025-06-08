# Basic Examples

This page provides a collection of basic examples to help you get started with rLLM.

## Simple Math Agent

This example shows how to create and use a basic math agent without training:

```python
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment

# Create a math agent
agent = MathAgent(
    model_path="agentica-org/DeepScaleR-1.5B-Preview",
    temperature=0.1
)

# Create an environment
env = SingleTurnEnvironment()

# Define a math problem
problem = {
    "data_source": "math_examples",
    "question": "If x + 5 = 12, what is the value of x?",
    "ground_truth": "7"
}

# Reset the environment with the problem
observation = env.reset(task=problem)

# Let the agent solve the problem
action = agent.act(observation)
print(f"Agent's answer: {action}")

# Evaluate the agent's solution
reward, _ = env.get_reward_and_next_obs(problem, action)
print(f"Reward: {reward}")
```

## Using Different Models

You can use different language models with rLLM:

```python
# Using OpenAI models
from rllm.agents import MathAgent

openai_agent = MathAgent(
    model_path="gpt-3.5-turbo",
    engine_name="openai",
    api_key="your-openai-api-key",
    temperature=0.1
)

# Using Anthropic models
anthropic_agent = MathAgent(
    model_path="claude-3-opus-20240229",
    engine_name="anthropic",
    api_key="your-anthropic-api-key",
    temperature=0.1
)

# Using local models
local_agent = MathAgent(
    model_path="/path/to/local/model",
    engine_name="verl",
    temperature=0.1
)
```

## Multi-Turn Interaction

This example demonstrates a multi-turn interaction with an agent:

```python
from rllm.agents import MathAgent
from rllm.environments.base import MultiTurnEnvironment

# Create a math agent
agent = MathAgent(
    model_path="agentica-org/DeepScaleR-1.5B-Preview",
    temperature=0.2
)

# Create a multi-turn environment
env = MultiTurnEnvironment(max_turns=3)

# Define a complex problem that benefits from multiple steps
problem = {
    "data_source": "math_examples",
    "initial_prompt": "Let's solve this step by step: Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3.",
    "ground_truth": "f'(x) = 3x^2 + 4x - 5"
}

# Reset the environment with the problem
observation = env.reset(task=problem)

# Interact for multiple turns
done = False
while not done:
    # Agent generates a response
    action = agent.act(observation)
    print(f"Agent: {action}")
    
    # Environment processes the action
    observation, reward, done, info = env.step(action)
    if not done:
        print(f"Environment: {observation['text']}")
    
# Print final reward
print(f"Final reward: {reward}")
```

## Batch Processing

This example shows how to process multiple tasks in parallel:

```python
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment
from rllm.engine import AsyncAgentExecutionEngine
import asyncio

# Create multiple problems
problems = [
    {
        "data_source": "math_examples",
        "question": "What is 15 + 27?",
        "ground_truth": "42"
    },
    {
        "data_source": "math_examples",
        "question": "What is 8 * 9?",
        "ground_truth": "72"
    },
    {
        "data_source": "math_examples",
        "question": "What is 100 / 4?",
        "ground_truth": "25"
    }
]

# Create multiple agents and environments
agents = [MathAgent(model_path="agentica-org/DeepScaleR-1.5B-Preview", temperature=0.1) for _ in range(len(problems))]
envs = [SingleTurnEnvironment(task=problem) for problem in problems]

# Create an execution engine
engine = AsyncAgentExecutionEngine(
    rollout_engine="verl",
    engine_name="verl",
    tokenizer=None,  # Will be initialized by the engine
    config={"agent": {"enable_thinking": True}},
    agents=agents,
    envs=envs,
    model_path="agentica-org/DeepScaleR-1.5B-Preview",
    max_steps=1,
    n_parallel_agents=len(problems)
)

# Run the trajectories in parallel
async def run_batch():
    tasks = [engine.run_agent_trajectory_async(idx, f"app_{idx}") for idx in range(len(problems))]
    return await asyncio.gather(*tasks)

# Execute batch processing
trajectories = asyncio.run(run_batch())

# Print results
for i, trajectory in enumerate(trajectories):
    print(f"Problem {i+1}: {problems[i]['question']}")
    print(f"Agent's answer: {trajectory[-1]['action']}")
    print(f"Reward: {trajectory[-1]['reward']}")
    print()
```

## Custom Reward Function

This example demonstrates how to create and use a custom reward function:

```python
from rllm.rewards import RewardResponse
from rllm.environments.base import SingleTurnEnvironment
from rllm.agents import MathAgent

# Define a custom reward function
def custom_math_reward(data_source, llm_solution, ground_truth, **kwargs):
    """
    A custom reward function that rewards correctness and explanations.
    """
    # Extract the final answer (simplified example)
    lines = llm_solution.strip().split('\n')
    final_line = lines[-1].lower()
    
    # Check if the solution contains the correct answer
    contains_correct_answer = ground_truth in final_line
    
    # Check if the solution includes explanation steps
    has_explanation = len(lines) > 2
    
    # Compute the reward
    reward = 0.0
    if contains_correct_answer:
        reward += 1.0  # Base reward for correctness
        
        # Additional reward for providing explanation
        if has_explanation:
            reward += 0.5
    
    return RewardResponse(
        reward=reward,
        metadata={
            "contains_correct_answer": contains_correct_answer,
            "has_explanation": has_explanation
        }
    )

# Create an environment with the custom reward function
env = SingleTurnEnvironment(reward_fn=custom_math_reward)

# Define a math problem
problem = {
    "data_source": "math_examples",
    "question": "What is the area of a circle with radius 4?",
    "ground_truth": "50.27"  # π × 4²
}

# Reset the environment with the problem
observation = env.reset(task=problem)

# Create and use a math agent
agent = MathAgent(
    model_path="agentica-org/DeepScaleR-1.5B-Preview",
    temperature=0.1
)

# Let the agent solve the problem
action = agent.act(observation)
print(f"Agent's answer: {action}")

# Evaluate the agent's solution with the custom reward
reward, _ = env.get_reward_and_next_obs(problem, action)
print(f"Reward: {reward}")
```

## Next Steps

- Explore more advanced examples for [Web Agents](web-agents.md), [Math Agents](math-agents.md), and [Code Agents](code-agents.md)
- Learn about [Training](../core-concepts/training.md) to improve agent performance 