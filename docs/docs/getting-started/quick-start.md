# Quick Start

This guide will help you get started with rLLM by walking through a simple example. We'll create and train a basic agent to solve a simple task.

## Basic Example: Single-Turn Math Solver

Let's create a simple agent that learns to solve math problems.

### 1. Define the Task

First, let's create a dataset of simple math problems:

```python
import json
from pathlib import Path

# Define a few simple math problems
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

# Create a directory for our dataset
Path("data/math_examples").mkdir(parents=True, exist_ok=True)

# Save the problems to a jsonl file
with open("data/math_examples/train.jsonl", "w") as f:
    for problem in problems:
        f.write(json.dumps(problem) + "\n")
```

### 2. Create a Custom Agent

Let's use the built-in Math Agent:

```python
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment
from rllm.train import AgentTrainer
from rllm.data import Dataset

# Create a dataset object
train_dataset = Dataset("data/math_examples/train.jsonl")

# Define a simple config
config = {
    "agent": {
        "model_path": "agentica-org/DeepScaleR-1.5B-Preview",  # Or your preferred model
        "temperature": 0.1
    },
    "train": {
        "num_epochs": 5,
        "learning_rate": 5e-5
    },
    "data": {
        "train_batch_size": 4,
        "val_batch_size": 4
    }
}

# Create a trainer
trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    config=config,
    train_dataset=train_dataset
)

# Start training
trainer.train()
```

### 3. Evaluate the Agent

After training, you can evaluate your agent on new math problems:

```python
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment

# Initialize the agent with the trained model
agent = MathAgent(model_path="path/to/your/trained/model", temperature=0.1)

# Initialize the environment
env = SingleTurnEnvironment()

# Define a new problem
new_problem = {
    "data_source": "math_examples",
    "question": "What is 12 * 12?",
    "ground_truth": "144"
}

# Reset the environment with the new problem
observation = env.reset(task=new_problem)

# Let the agent solve the problem
action = agent.act(observation)
print(f"Agent's answer: {action}")

# Get the reward
reward, _ = env.get_reward_and_next_obs(new_problem, action)
print(f"Reward: {reward}")
```

## Next Steps

This is a very simple example to get you started. rLLM supports much more complex scenarios:

- **Multi-turn interactions**: For tasks requiring dialogue or sequential decision-making
- **Web browsing agents**: For interacting with websites
- **Code generation**: For programming tasks
- **Tool use**: For agents that can use external tools

Check out the more advanced examples in the [Examples](../examples/basic.md) section to learn more about these capabilities. 