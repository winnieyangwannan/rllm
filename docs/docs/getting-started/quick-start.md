# Quick Start with rLLM

This guide walks you through using rLLM to build AI agents with tool usage capabilities. We'll use the **math tool agent** example to demonstrate the complete workflow from dataset preparation through model training.

## Overview

In this tutorial, you'll create a math reasoning agent that can:

- Access a Python interpreter to solve mathematical problems
- Perform step-by-step reasoning with interleaved tool usage
- Learn and improve its math problem solving ability through reinforcement learning

The example uses:

- **Base Model**: Qwen3-4B
- **Training Data**: DeepScaleR-Preview-Math dataset 
- **Evaluation Data**: AIME 2024 mathematics competition problems
- **Tools**: Python interpreter for mathematical computations

## Prerequisites

Before starting, ensure you have:

1. **rLLM Installation**: Follow the [installation guide](../installation.md)
2. **GPU Requirements**: At least 1 GPU with 16GB+ memory for inference, 8+ GPUs for training
3. **Model Server**: We'll use vLLM or SGLang to serve the base model

## Step 1: Dataset Preparation

rLLM's `DatasetRegistry` provides a centralized way to manage datasets. Let's prepare the math datasets:

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

    # Standardize data format for rLLM
    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],        # The math problem to solve
            "ground_truth": example["answer"],     # The correct answer
            "data_source": "math",                 # Dataset identifier
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    # Register datasets with rLLM's registry for easy access
    train_dataset = DatasetRegistry.register_dataset(
        "deepscaler_math", train_dataset, "train"
    )
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    
    return train_dataset, test_dataset
```

This registers the training dataset `deepscaler_math` and the testing dataset `aime2024`. Under the hood, rLLM stores the processed data as parquet files in a format suitable for both inference and training. Later, you can easily load the registered datasets using `DatasetRegistry.load_dataset`.

**Run the preparation script:**
```bash
cd examples/math_tool
python prepare_math_data.py
```

## Step 2: Model Server Setup

rLLM requires a model server for inference. Choose one of these options:

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
    env_class=ToolEnvironment,                # Environment that provides tool access
    agent_args=agent_args,
    env_args=env_args,
    engine_name="openai",                     # Use OpenAI-compatible API
    rollout_engine_args={                     # Connection to model server
        "base_url": "http://localhost:30000/v1", 
        "api_key": "None"
    },
    tokenizer=tokenizer,
    sampling_params=sampling_params,
    max_response_length=16384,                # Maximum tokens in response
    max_prompt_length=2048,                   # Maximum tokens in prompt
    n_parallel_agents=64,                     # Parallel execution for efficiency
)

# Load test dataset and run inference
test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
tasks = test_dataset.repeat(n=8)              # Repeat for Pass@K evaluation

# Execute tasks asynchronously
results = asyncio.run(engine.execute_tasks(tasks))
compute_pass_at_k(results)                    # Compute Pass@1, Pass@8 metrics
```

**Run the inference script:**
```bash
cd examples/math_tool
python run_math_with_tool.py
```

The script above configures a `ToolAgent` from rLLM with access to the `python` tool for solving math problems in AIME2024, and a `ToolEnvironment` for handling Python tool calls and returning results. 

The `AsyncAgentExecutionEngine` orchestrates the interaction between the `ToolAgent` and `ToolEnvironment`. The `execute_tasks` function launches 64 agent-environment pairs in parallel (`n_parallel_agents=64`) for rollout generation and returns results after all problems from the AIME2024 dataset are processed. Finally, the Pass@1 and Pass@K metrics for AIME are computed and printed. 

## Step 4: Agent Training with GRPO

Training improves the agent's ability to use tools effectively. rLLM uses verl as its training backend, which supports training language models with GRPO and various other RL algorithms.

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

The script above launches an RL training job for our ToolAgent, using `deepscaler_math` as the training set and `aime2024` as the test set. Under the hood, rLLM handles agent trajectory generation using our `AsyncAgentExecutionEngine` and transforms the trajectories into `verl`'s format for model training using FSDP or Megatron. The training process works as follows:

1. **Rollout Generation**: A batch of data is passed to `AsyncAgentExecutionEngine`, which launches multiple agent-environment pairs in parallel to process the batch. The engine returns all trajectories along with rewards computed by the environment.
2. **Transform Trajectories**: Agent trajectories are transformed into the corresponding format for our training backend `verl`. 
3. **Advantage Calculation with GRPO**: `verl` uses GRPO for advantage calculation.
4. **Model Update**: `verl` updates the model parameters to increase the probability of successful actions. The updated model is then used to generate trajectories for the next batch of data.

### Key rLLM Components in This Example

| Component | Purpose | Example Usage |
|-----------|---------|---------------|
| `ToolAgent` | Agent with tool usage capabilities | Reasoning + Python execution |
| `ToolEnvironment` | Safe tool execution environment | Sandboxed Python interpreter |
| `DatasetRegistry` | Centralized dataset management | Load/register math datasets |
| `AsyncAgentExecutionEngine` | Parallel agent execution | Efficient batch inference |
| `AgentTrainer` | RL training orchestration | PPO-based agent improvement |

## Next Steps

Congratulations! You've successfully used rLLM to run and train a ToolAgent for math problem solving. For a deeper dive into rLLM's main components, check out [Core Concepts in rLLM](../core-concepts/overview.md).