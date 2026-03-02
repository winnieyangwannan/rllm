# Example: External Agent Plugin

This example shows how to build a custom agent project that plugs into rLLM's CLI for both evaluation and training.

The "concierge agent" recommends restaurants based on a user query. A custom "relevance evaluator" checks whether the response mentions the expected cuisine.

## Project Structure

```
examples/cli/agent_plugin/
├── pyproject.toml                    # Package config with entry points
├── concierge_agent/
│   ├── __init__.py
│   ├── agent.py                      # ConciergeAgent (AgentFlow protocol)
│   └── evaluator.py                  # RelevanceEvaluator
├── data/
│   └── concierge_test.json           # 10 sample restaurant queries
└── tests/
    ├── __init__.py
    └── test_agent.py                 # Unit tests (mocked, no LLM needed)
```

## Setup

```bash
# Install rLLM (if not already)
uv pip install -e .

# Install the plugin in dev mode
uv pip install -e examples/cli/agent_plugin/
```

## Verify Plugin Discovery

```bash
# The concierge agent should appear with source "plugin (concierge-agent)"
rllm agent list
```

## Register a Dataset

```bash
rllm dataset register concierge_test \
    --file examples/cli/agent_plugin/data/concierge_test.json \
    --split test --category qa --description "Restaurant concierge test set"

# Verify
rllm dataset list
rllm dataset inspect concierge_test --split test -n 3
```

## Run Unit Tests (no LLM needed)

```bash
pytest examples/cli/agent_plugin/tests/ -v
```

## Evaluate (requires a running LLM endpoint)

```bash
rllm eval concierge_test --agent concierge --evaluator relevance
```

## Train with RL (requires tinker backend)

```bash
rllm train concierge_test --agent concierge --evaluator relevance \
    --model Qwen/Qwen3-8B --group-size 8 --batch-size 32 --epochs 3
```

## Alternative: Import Paths Instead of Entry Points

You can skip entry-point registration and use `module:object` paths directly:

```bash
rllm eval concierge_test \
    --agent concierge_agent.agent:concierge_agent \
    --evaluator concierge_agent.evaluator:RelevanceEvaluator \
    --base-url http://localhost:8000/v1 --model gpt-4o
```

## How It Works

The two key integration points are in `pyproject.toml`:

```toml
[project.entry-points."rllm.agents"]
concierge = "concierge_agent.agent:concierge_agent"

[project.entry-points."rllm.evaluators"]
relevance = "concierge_agent.evaluator:RelevanceEvaluator"
```

After `pip install -e .`, rLLM discovers these via Python's `importlib.metadata.entry_points` API. The agent must implement `run(task, config) -> Episode` (the `AgentFlow` protocol) and the evaluator must implement `evaluate(task, episode) -> EvalOutput` (the `Evaluator` protocol).
