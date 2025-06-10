# Quick Start with RLLM

This guide walks you through using RLLM to build AI agents with tool usage capabilities. We'll use the **math tool agent** example to demonstrate the complete workflow from dataset preparation through model training.

## Overview

In this tutorial, we'll create a math reasoning agent that can:
- Access a Python interpreter to solve mathematical problems
- Perform step-by-step reasoning with tool usage
- Learn through reinforcement learning from human feedback (RLHF)

The example uses:
- **Base Model**: Qwen3-4B (a capable instruction-following LLM)
- **Training Data**: DeepScaleR-Math dataset 
- **Evaluation Data**: AIME 2024 mathematics competition problems
- **Tools**: Python interpreter for mathematical computations

## Prerequisites

1. **Install rLLM**: Follow the [installation guide](../installation.md)
2. **GPU Requirements**: At least 1 GPU with 16GB+ memory for inference, 8+ GPUs for training
3. **Model Server**: We'll use vLLM or SGLang to serve the base model

## Step 1: Dataset Preparation

RLLM's `DatasetRegistry` provides a centralized way to manage datasets. Let's examine how to prepare math datasets:

```python
from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry

def prepare_math_data():
    # Load training data from DeepScaleR (math reasoning dataset)
    train_dataset = load_dataset(
        "agentica-org/DeepScaleR-Preview-Dataset", split="train"
    )
    
    # Load test data from AIME 2024 (competition math problems)
    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    # Standardize data format for RLLM
    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],        # The math problem to solve
            "ground_truth": example["answer"],     # The correct answer
            "data_source": "math",                 # Dataset identifier
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    # Register datasets with RLLM's registry for easy access
    train_dataset = DatasetRegistry.register_dataset(
        "deepscaler_math", train_dataset, "train"
    )
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    
    return train_dataset, test_dataset
```

**Run the preparation script:**
```bash
cd examples/math_tool
python prepare_math_data.py
```

**Key Concepts:**
- **DatasetRegistry**: Centralized dataset management for consistent access across training and inference
- **Preprocessing**: Standardizing data format ensures compatibility with RLLM's agent framework
- **Ground Truth**: Reference answers enable reward computation during training

## Step 2: Model Server Setup

RLLM requires a model server for inference. Choose one of these options:

### Option A: vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

### Option B: SGLang Server  
```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \
    --dp-size 1 \
    --dtype bfloat16
```

The server provides an OpenAI-compatible API at `http://localhost:30000/v1`.

## Step 3: Model Inference

Now let's run inference to see how agents solve math problems using tools:

```python
import asyncio
from transformers import AutoTokenizer
from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k

# Configure the model and tokenizer
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Agent configuration - what tools can the agent use?
agent_args = {
    "tools": ["python"],           # Python interpreter access
    "parser_name": "qwen"         # Parser for Qwen model outputs
}

# Environment configuration - how should agent interactions be handled?
env_args = {
    "tools": ["python"],          # Available tools in environment
    "reward_fn": math_reward_fn,  # Function to compute rewards
}

# Sampling parameters for model generation
sampling_params = {
    "temperature": 0.6,           # Creativity vs consistency balance
    "top_p": 0.95,               # Nucleus sampling threshold
    "model": model_name
}

# Create the execution engine
engine = AsyncAgentExecutionEngine(
    agent_class=ToolAgent,                    # Use ToolAgent for tool-enabled reasoning
    agent_args=agent_args,
    env_class=ToolEnvironment,                # Environment that provides tool access
    env_args=env_args,
    rollout_engine=None,
    engine_name="openai",                     # Use OpenAI-compatible API
    tokenizer=tokenizer,
    sampling_params=sampling_params,
    rollout_engine_args={                     # Connection to model server
        "base_url": "http://localhost:30000/v1", 
        "api_key": "None"
    },
    max_response_length=16384,                # Maximum tokens in response
    max_prompt_length=2048,                   # Maximum tokens in prompt
    n_parallel_agents=64,                     # Parallel execution for efficiency
)

# Load test dataset and run inference
test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
tasks = test_dataset.repeat(n=4)              # Repeat for Pass@K evaluation

# Execute tasks asynchronously
results = asyncio.run(engine.execute_tasks(tasks))
compute_pass_at_k(results)                    # Compute Pass@1, Pass@4 metrics
```

**Run the inference script:**
```bash
cd examples/math_tool
python run_math_with_tool.py
```

**Key Concepts:**
- **ToolAgent**: An agent class that can use external tools (Python interpreter) for reasoning
- **ToolEnvironment**: Provides a safe sandbox for tool execution and reward computation
- **AsyncAgentExecutionEngine**: Efficiently handles parallel agent execution for scalability
- **Pass@K Evaluation**: Measures success rate when allowing K attempts per problem

## Step 4: Agent Training with RLHF

Training improves the agent's ability to use tools effectively. RLLM uses PPO (Proximal Policy Optimization) for this:

```python
import hydra
from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.train.agent_trainer import AgentTrainer
from rllm.rewards.reward_fn import math_reward_fn

@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Load prepared datasets
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    # Same configuration as inference
    agent_args = {"tools": ["python"], "parser_name": "qwen"}
    env_args = {"tools": ["python"], "reward_fn": math_reward_fn}
    
    # Create trainer with datasets and configuration
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,                        # Hydra configuration for training
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    # Start training process
    trainer.train()
```

**Run the training script:**
```bash
cd examples/math_tool
bash train_math_with_tool.sh
```

**Key Training Configuration:**
```bash
# Key hyperparameters from train_math_with_tool.sh
algorithm.adv_estimator=grpo                 # Advantage estimation method
data.train_batch_size=32                     # Training batch size
data.max_response_length=8192                # Max response length during training
actor_rollout_ref.actor.optim.lr=1e-6       # Learning rate
actor_rollout_ref.rollout.temperature=0.6    # Generation temperature
trainer.total_epochs=100                      # Number of training epochs
```

**Key Concepts:**
- **PPO Training**: Policy optimization that balances exploration vs exploitation
- **Reward Function**: `math_reward_fn` evaluates correctness of mathematical solutions
- **GRPO**: Group-relative policy optimization for improved training stability
- **Hybrid Engine**: Combines training and inference engines for efficiency

## Understanding the Complete Workflow

### 1. Agent-Environment Interaction Loop
```
Problem → Agent → Tool Call → Environment → Reward → Agent Update
```

- **Agent** receives a math problem and generates reasoning steps
- **Tool calls** execute Python code to perform calculations  
- **Environment** provides safe execution and computes rewards
- **Rewards** guide learning to improve problem-solving strategies

### 2. Key RLLM Components

| Component | Purpose | Example Usage |
|-----------|---------|---------------|
| `ToolAgent` | Agent with tool usage capabilities | Reasoning + Python execution |
| `ToolEnvironment` | Safe tool execution environment | Sandboxed Python interpreter |
| `DatasetRegistry` | Centralized dataset management | Load/register math datasets |
| `AsyncAgentExecutionEngine` | Parallel agent execution | Efficient batch inference |
| `AgentTrainer` | RL training orchestration | PPO-based agent improvement |

### 3. Training Process Details

1. **Rollout Phase**: Agents attempt problems and generate solution attempts
2. **Reward Computation**: Solutions are evaluated against ground truth answers
3. **Advantage Estimation**: Determine which actions led to better outcomes
4. **Policy Update**: Adjust model parameters to increase probability of successful actions
5. **Validation**: Test improved agent on held-out problems

## Next Steps

Now that you understand the basics, you can:

1. **Modify the Agent**: Add new tools or change reasoning strategies
2. **Custom Datasets**: Prepare your own datasets for specific domains
3. **Hyperparameter Tuning**: Adjust training parameters for better performance
4. **Multi-Agent Systems**: Explore collaborative agent scenarios

For more advanced topics, see:
- [Agent Architecture Guide](../concepts/agents.md)
- [Environment Configuration](../concepts/environments.md)  
- [Training Configuration](../training/configuration.md)
- [Custom Tools Development](../tools/custom-tools.md)

## Troubleshooting

**Common Issues:**
- **OOM Errors**: Reduce batch size or sequence length
- **Slow Training**: Increase `n_parallel_agents` or use more GPUs
- **Poor Performance**: Check reward function and dataset quality
- **Connection Errors**: Verify model server is running on correct port

**Performance Tips:**
- Use `hybrid_engine=True` for better memory efficiency
- Enable gradient checkpointing for large models
- Adjust `gpu_memory_utilization` based on your hardware
