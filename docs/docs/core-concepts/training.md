# Training

Training in rLLM uses reinforcement learning algorithms to update agent policies based on rewards. This page explains the training architecture, available algorithms, and how to configure and run training jobs.

## Training Architecture

The training system in rLLM consists of several components:

1. **AgentTrainer**: A high-level interface for training agents with specific environments
2. **PPO Trainer**: Implementation of the Proximal Policy Optimization algorithm
3. **Rollout Engine**: Manages the collection of trajectories from agent-environment interactions
4. **Execution Engine**: Orchestrates agent-environment interactions and computes rewards

## Training Process

The typical training process in rLLM follows these steps:

1. **Initialization**: Set up agents, environments, and the training configuration
2. **Rollout**: Collect trajectories by having agents interact with environments
3. **Advantage Estimation**: Compute advantages based on rewards
4. **Policy Update**: Update the agent policy using the PPO algorithm
5. **Evaluation**: Evaluate the updated policy on validation tasks
6. **Iteration**: Repeat steps 2-5 until convergence or a maximum number of epochs

## Using AgentTrainer

The `AgentTrainer` class provides a high-level interface for training agents:

```python
from rllm.train import AgentTrainer
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment
from rllm.data import Dataset

# Create datasets
train_dataset = Dataset("data/math_examples/train.jsonl")
val_dataset = Dataset("data/math_examples/val.jsonl")

# Create a configuration
config = {
    "agent": {
        "model_path": "your/model/path",
        "temperature": 0.1
    },
    "train": {
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "max_grad_norm": 0.5
    },
    "data": {
        "train_batch_size": 16,
        "val_batch_size": 16
    }
}

# Create a trainer
trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Start training
trainer.train()
```

## Advanced Configuration

rLLM provides extensive configuration options for training:

```python
config = {
    "agent": {
        "model_path": "your/model/path",
        "temperature": 0.1,
        "max_response_length": 1024,
        "engine_name": "openai",  # or "verl", etc.
        "enable_thinking": True,  # Enable chain-of-thought
    },
    "train": {
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "max_grad_norm": 0.5,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 100,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "normalise_advantages": True,
    },
    "data": {
        "train_batch_size": 16,
        "val_batch_size": 16,
        "train_files": ["data/train.jsonl"],
        "val_files": ["data/val.jsonl"],
    },
    "evaluation": {
        "eval_freq": 1,  # Evaluate every N epochs
        "save_freq": 1,  # Save checkpoint every N epochs
        "log_freq": 10,  # Log metrics every N steps
    },
    "logging": {
        "wandb": True,  # Enable Weights & Biases logging
        "project_name": "rllm-project",
        "run_name": "math-agent-training",
    },
    "resources": {
        "num_workers": 4,  # Number of parallel workers
        "use_gpu": True,
        "precision": "bf16",  # or "fp16", "fp32"
    }
}
```

## Distributed Training

rLLM supports distributed training across multiple machines using Ray:

```python
import ray
from rllm.train import AgentTrainer

# Initialize Ray
ray.init(address="auto")

# Create a trainer with distributed settings
config = {
    # ... other configuration options ...
    "resources": {
        "num_workers": 8,
        "num_gpus": 4,
        "num_cpus_per_worker": 2,
        "num_gpus_per_worker": 0.5,
    },
    "distributed": {
        "enabled": True,
        "backend": "ray",
        "world_size": 4,
    }
}

trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

trainer.train()
```

## Monitoring Training

rLLM integrates with Weights & Biases for training monitoring:

```python
config = {
    # ... other configuration options ...
    "logging": {
        "wandb": True,
        "project_name": "rllm-project",
        "run_name": "math-agent-training",
        "log_freq": 10,
        "log_gradient_histograms": True,
        "log_model_parameters": True,
    }
}
```

## Saving and Loading Models

rLLM provides utilities for saving and loading trained models:

```python
from rllm.train import AgentTrainer
from rllm.agents import MathAgent
from rllm.environments.base import SingleTurnEnvironment

# Train an agent
trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=SingleTurnEnvironment,
    config=config
)
trainer.train()

# Save the trained model
trainer.save_model("path/to/save/model")

# Load a trained model for inference
agent = MathAgent.from_pretrained("path/to/saved/model")
```

## Next Steps

- Explore [Examples](../examples/basic.md) of training agents for specific tasks
- Learn about the [API Reference](../api/trainer.md) for detailed training documentation 