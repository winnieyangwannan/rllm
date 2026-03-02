# Example: Python API Registration

This example shows how to register an agent, evaluator, and dataset via the Python API, then use them from the rLLM CLI.

Compare with [`../agent_plugin/`](../agent_plugin/) which uses `pyproject.toml` entry points for the same result.

## Project Structure

```
examples/cli/agent_python_api/
├── agent.py        # ConciergeAgent (AgentFlow protocol)
├── evaluator.py    # RelevanceEvaluator
├── data.json       # Sample dataset (10 restaurant queries)
├── register.py     # One-time registration script
├── evaluate.py     # Evaluation via Python API
└── train.py        # Training via Python API
```

## Key Difference from the Plugin Example

| | Plugin (`agent_plugin/`) | Python API (this example) |
|---|---|---|
| Agent registration | `pyproject.toml` entry points + `pip install` | `register_agent()` in a Python script |
| Evaluator registration | `pyproject.toml` entry points + `pip install` | `register_evaluator()` in a Python script |
| Dataset registration | `rllm dataset register` CLI | `DatasetRegistry.register_dataset()` |
| After registration | Works with `rllm` CLI | Also works with `rllm` CLI |

## Setup

```bash
# From the rllm root directory
uv pip install -e .
```

## Step 1: Register Everything

```bash
cd examples/cli/agent_python_api
python register.py
```

This persists to `~/.rllm/`:
- `~/.rllm/agents.json` — maps `"concierge"` to `"concierge_agent.agent:ConciergeAgent"`
- `~/.rllm/evaluators.json` — maps `"relevance"` to `"concierge_agent.evaluator:RelevanceEvaluator"`
- `~/.rllm/datasets/concierge/` — train and test splits

## Step 2: Use from the CLI

After registration, the agent and evaluator are available by name in the rllm CLI — from any terminal, any process:

```bash
# Verify registration
rllm agent list                     # shows "concierge" with source "registered"
rllm dataset list                   # shows "concierge" dataset

# Evaluate
rllm eval concierge --agent concierge --evaluator relevance \
    --base-url http://localhost:8000/v1 --model gpt-4o

# Train
rllm train concierge --agent concierge --evaluator relevance \
    --model Qwen/Qwen3-8B
```

## Or Use from Python Directly

```bash
python evaluate.py --base-url http://localhost:8000/v1 --model gpt-4o
python train.py --model Qwen/Qwen3-8B
```

## How It Works

### `register_agent()` / `register_evaluator()`

These functions persist an import path to `~/.rllm/agents.json` or `~/.rllm/evaluators.json`. You can pass a string, a class, or an instance:

```python
from rllm.experimental.eval.agent_loader import register_agent
from rllm.experimental.eval.evaluator_loader import register_evaluator

# Any of these forms work:
register_agent("concierge", "my_agent.agent:ConciergeAgent")  # import path string
register_agent("concierge", ConciergeAgent)                    # class (path auto-derived)
register_agent("concierge", ConciergeAgent())                  # instance (path auto-derived)

register_evaluator("relevance", "my_eval:RelevanceEvaluator")
```

After registration, `load_agent("concierge")` works from any process. The lookup order is:

1. **User registry** (`~/.rllm/agents.json`) — from `register_agent()`
2. **Import path** (`module:object`) — explicit colon syntax
3. **Built-in catalog** (`registry/agents.json`) — ships with rllm
4. **Entry points** (`rllm.agents` group) — from `pyproject.toml` plugins

### `DatasetRegistry.register_dataset()`

```python
from rllm.data import Dataset, DatasetRegistry

ds = Dataset.load_data("data.json")
DatasetRegistry.register_dataset("concierge", ds.data, split="test", category="qa")
```

### Unregistration

```python
from rllm.experimental.eval.agent_loader import unregister_agent
from rllm.experimental.eval.evaluator_loader import unregister_evaluator

unregister_agent("concierge")
unregister_evaluator("relevance")
```
