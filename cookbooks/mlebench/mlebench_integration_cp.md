## MLE-Bench Implementation Plan for rLLM

This section documents the plan to port MLE-bench support from AMAIA to rLLM, reusing AMAIA's AgentBox backend.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    rLLM                                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐    │
│   │  @rllm.rollout       │     │  @rllm.evaluator     │     │  AgentTrainer    │    │
│   │  mle_bench_agent()   │────▶│  mle_bench_eval()   │────▶│  (tinker/verl)   │    │
│   └──────────────────────┘     └──────────────────────┘     └──────────────────┘    │
│            │                              │                                          │
│            ▼                              ▼                                          │
│   ┌──────────────────────┐     ┌──────────────────────┐                             │
│   │  OpenAI Client       │     │  MLE-Bench Grading   │                             │
│   │  (Model Gateway)     │     │  (percentile calc)   │                             │
│   └──────────────────────┘     └──────────────────────┘                             │
│            │                                                                         │
│            ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐               │
│   │                    AMAIA AgentBox Backend                        │               │
│   │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │               │
│   │  │ AgentBoxManager │  │ ContainerConfig  │  │ Shell/Notebook │  │               │
│   │  │ (Container Pool)│  │ (H200 + Data)    │  │ Execution      │  │               │
│   │  └─────────────────┘  └──────────────────┘  └────────────────┘  │               │
│   └─────────────────────────────────────────────────────────────────┘               │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Mapping

| AMAIA Component | rLLM Equivalent | Notes |
|-----------------|-----------------|-------|
| `MLEBenchBashEnv` | `mle_bench_agent()` | Rollout function with `@rllm.rollout` |
| `MLEBenchBashState` | Internal state in agent | Track turn count, tokens, solution |
| `AgentBoxBackend` | **Reuse directly** | Import from AMAIA |
| `agentbox_rpc_eval()` | `mle_bench_evaluator()` | Wrap in `@rllm.evaluator` |
| `Transition` | `Step` | Map fields accordingly |
| `Trajectory` (AMAIA) | `Trajectory` + `Episode` (rLLM) | Episode contains single Trajectory |
| `PromptSet` | Config/prompts module | Same structure |

### File Structure (Proposed)

```
cookbooks/mle_bench/
├── __init__.py
├── pyproject.toml
├── README.md
├── mle_bench_flow.py          # @rllm.rollout agent
├── evaluator.py               # @rllm.evaluator
├── agentbox_wrapper.py        # Thin wrapper around AMAIA's AgentBoxBackend
├── grading.py                 # MLE-bench specific grading (percentile calc)
├── prompts/
│   ├── __init__.py
│   ├── system.py              # System prompts (port from AMAIA)
│   ├── instance.py            # Instance prompt template
│   └── common.py              # Action/observation templates
├── configs/
│   ├── default.yaml           # Default training config
│   └── eval.yaml              # Evaluation config
├── train.py                   # Training entrypoint
├── test.py                    # Unit tests
└── data/
    └── mle_bench_tasks.jsonl  # Task definitions (or fetch from registry)
```

### Component Details

#### 1. Agent Rollout (`mle_bench_flow.py`)

```python
"""MLE-Bench AgentFlow — autonomous ML engineering agent."""

from __future__ import annotations
import time
from dataclasses import dataclass
from openai import OpenAI

import rllm
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Step, Trajectory

from .agentbox_wrapper import AgentBoxWrapper
from .prompts import get_prompt_set, format_system_prompt, format_instance_prompt


@dataclass
class MLEBenchConfig:
    """Configuration for MLE-Bench agent."""
    agentbox_manager_uri: str
    max_turns: int = 128
    session_timeout: float = 360.0      # Per-bash-call timeout
    eval_timeout: int = 300             # Final evaluation timeout
    rollout_timeout: int = 32400        # Total rollout time budget (9 hours)
    context_size: int = 131072
    think: bool = True
    message_format: str = "xml"         # "xml" or "json"


@rllm.rollout(name="mle-bench")
def mle_bench_agent(task: Task, config: AgentConfig) -> Episode:
    """
    MLE-Bench agent: autonomous ML engineering on Kaggle-style competitions.
    
    The agent:
    1. Reads the competition description and data structure
    2. Iteratively writes/edits code using bash, edit, create tools
    3. Trains a model on train data
    4. Generates predictions on test data
    5. Submits final solution.py → writes submission.csv
    
    Returns Episode with single Trajectory containing all Steps.
    """
    mle_config = MLEBenchConfig(**task.data.get("config", {}))
    
    # Initialize LLM client (Model Gateway handles trace capture)
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    
    # Initialize AgentBox container
    agentbox = AgentBoxWrapper(
        task_id=task.data["task_id"],
        benchmark="mlebench",
        manager_uri=mle_config.agentbox_manager_uri,
        session_timeout=mle_config.session_timeout,
    )
    
    try:
        # Build initial prompt
        prompt_set = get_prompt_set(model="gpt5", think=mle_config.think)
        system_prompt = format_system_prompt(
            prompt_set.system_prompt,
            timeout_min=int(mle_config.session_timeout / 60),
            context_size=mle_config.context_size,
            max_turns=mle_config.max_turns,
            eval_timeout_hrs=int(mle_config.eval_timeout / 3600),
        )
        data_info = agentbox.gather_data_info()
        instance_prompt = format_instance_prompt(
            prompt_set.instance_prompt,
            task_description=task.data["task_description"],
            data_info=data_info,
        )
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        
        steps = []
        start_time = time.time()
        pred_solution = None
        
        for turn in range(mle_config.max_turns):
            # Check rollout timeout
            elapsed = time.time() - start_time
            if elapsed >= mle_config.rollout_timeout:
                break
            
            # Get LLM response
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=1.0,
                max_tokens=mle_config.context_size // 4,
            )
            assistant_content = response.choices[0].message.content or ""
            
            # Parse tool call from response
            tool_name, tool_input, is_terminal = parse_tool_call(
                assistant_content, 
                message_format=mle_config.message_format
            )
            
            if tool_name is None:
                # No tool call - add thinking-only prompt
                observation = get_thinking_only_prompt()
            else:
                # Execute tool via AgentBox
                observation, is_terminal, pred_solution = agentbox.execute_tool(
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
            
            # Record step
            steps.append(Step(
                input=messages[-1]["content"],
                output=assistant_content,
                action=tool_name,
                metadata={
                    "turn": turn,
                    "tool_input": tool_input,
                    "observation": observation,
                },
            ))
            
            # Update conversation
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": observation})
            
            if is_terminal:
                break
        
        return Episode(
            trajectories=[
                Trajectory(
                    name="mle-agent",
                    steps=steps,
                    metadata={
                        "task_id": task.data["task_id"],
                        "pred_solution": pred_solution,
                        "total_turns": len(steps),
                        "elapsed_time": time.time() - start_time,
                    },
                )
            ],
            artifacts={"pred_solution": pred_solution},
        )
    
    finally:
        agentbox.cleanup()
```

#### 2. AgentBox Wrapper (`agentbox_wrapper.py`)

```python
"""Thin wrapper around AMAIA's AgentBox backend for rLLM."""

from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass

# Import AMAIA's AgentBox components
import sys
sys.path.insert(0, "/home/winnieyangwn/amaia-collab")
from apps.sea.envs.envs.mle_bench.agentbox_backend import (
    AgentBoxBackend,
    AgentBoxConfig,
    MLE_BENCH_DATA_DIR,
    AIRA_SUPERIMAGE_DIR,
    AIRA_SUPERIMAGE_VERSION,
    AIRA_SUPERIMAGE_OVERLAY,
)
from apps.rl.swerl.tools import make_bash, ToolType

logger = logging.getLogger(__name__)


@dataclass
class AgentBoxWrapper:
    """Wrapper around AMAIA's AgentBoxBackend for rLLM usage."""
    
    task_id: str
    benchmark: str = "mlebench"
    manager_uri: str = ""
    session_timeout: float = 360.0
    
    def __post_init__(self):
        """Initialize the AgentBox container."""
        # Build tools dict
        tools_dict: dict[str, ToolType] = {
            "bash": make_bash,
        }
        
        config = AgentBoxConfig(
            task=self.task_id,
            benchmark=self.benchmark,
            start_script="",
            plugin_root="/path/to/plugins",  # Configure as needed
            bind_target="/workspace/plugins",
            tools=tools_dict,
            plugin_names=["edit", "create"],
            manager_uri=self.manager_uri,
        )
        self.backend = AgentBoxBackend(config)
    
    def gather_data_info(self) -> str:
        """Gather data structure info from container."""
        command = '''cd /root/data && \\
echo "=== DATA STRUCTURE ===" && ls -sh && \\
echo -e "\\n=== CSV ROW COUNTS ===" && wc -l *.csv 2>/dev/null && \\
echo -e "\\n=== SAMPLE SUBMISSION FORMAT ===" && head -3 sample_submission.csv'''
        
        result = self.backend.run_bash(command, timeout=30.0)
        return result.output if result.status == "success" else "Data info unavailable"
    
    def execute_tool(
        self, tool_name: str, tool_input: str
    ) -> tuple[str, bool, str | None]:
        """
        Execute a tool via AgentBox.
        
        Returns:
            (observation, is_terminal, pred_solution)
        """
        is_terminal = False
        pred_solution = None
        
        if tool_name == "bash":
            result = self.backend.run_bash(tool_input, timeout=self.session_timeout)
            observation = format_bash_output(result)
        
        elif tool_name == "submit":
            # Read solution file and mark as terminal
            result = self.backend.run_bash(f"cat {tool_input}", timeout=30.0)
            pred_solution = result.output if result.status == "success" else None
            observation = "Solution submitted for evaluation."
            is_terminal = True
        
        elif tool_name in ["edit", "create"]:
            result = self.backend.apply_tool(tool_name, tool_input, timeout=30.0)
            observation = result.output
        
        else:
            observation = f"Unknown tool: {tool_name}"
        
        return observation, is_terminal, pred_solution
    
    def put_file(self, local_path: Path, container_path: Path) -> None:
        """Upload file to container."""
        self.backend.put_file(local_path, container_path)
    
    def fetch_file(self, container_path: Path, local_path: Path) -> bool:
        """Fetch file from container."""
        return self.backend.fetch_file(container_path, local_path)
    
    def cleanup(self) -> None:
        """Clean up container resources."""
        if hasattr(self.backend, 'container'):
            try:
                self.backend.container.stop()
            except Exception as e:
                logger.warning(f"Failed to stop container: {e}")
```

#### 3. Evaluator (`evaluator.py`)

```python
"""MLE-Bench evaluator: scores submissions using percentile ranking."""

from __future__ import annotations
import tempfile
from pathlib import Path
import logging

import rllm
from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.types import Episode

# Import AMAIA's evaluation functions
import sys
sys.path.insert(0, "/home/winnieyangwn/amaia-collab")
from apps.sea.envs.envs.mle_bench.evaluation import (
    agentbox_rpc_eval,
    MLEBenchTestResult,
    get_rank_and_percentile,
)
from apps.sea.envs.envs.mle_bench.agentbox_backend import (
    AgentBoxBackend,
    AgentBoxConfig,
    MLE_BENCH_DATA_DIR,
)

logger = logging.getLogger(__name__)


@rllm.evaluator
def mle_bench_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """
    Evaluate MLE-bench submission.
    
    Scoring:
    - Extracts pred_solution from episode artifacts
    - Runs solution in AgentBox container (if code mode)
    - Validates submission.csv format
    - Computes percentile rank against public leaderboard
    
    Reward:
    - 1.0 for valid submission with percentile >= 0.5 (median or better)
    - percentile value for valid submissions below median
    - 0.0 for invalid submissions
    """
    task_id = task.get("task_id") or task.get("instance_id")
    pred_solution = episode.artifacts.get("pred_solution")
    
    # Get trajectory metadata
    traj = episode.trajectories[0] if episode.trajectories else None
    
    if not pred_solution:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[
                Signal(name="percentile", value=0.0),
                Signal(name="valid_submission", value=0.0),
                Signal(name="submission_provided", value=0.0),
            ],
        )
    
    # Re-create AgentBox for evaluation (or reuse from episode if available)
    config = AgentBoxConfig(
        task=task_id,
        benchmark="mlebench",
        start_script="",
        plugin_root="",
        bind_target="",
        tools={},
        plugin_names=[],
        manager_uri=task.get("agentbox_manager_uri", ""),
    )
    backend = AgentBoxBackend(config)
    
    try:
        test_result: MLEBenchTestResult = agentbox_rpc_eval(
            backend=backend,
            eval_timeout=task.get("eval_timeout", 300),
            task=task_id,
            pred_solution=pred_solution,
            eval_execution_mode=task.get("eval_execution_mode", "bash"),
            submit_file=task.get("submit_file", "code"),
        )
        
        percentile = test_result.percentile or 0.0
        is_valid = test_result.outcome == "pass"
        
        # Reward: percentile for valid submissions, 0 otherwise
        reward = percentile if is_valid else 0.0
        
        # Trajectory-level reward for GRPO
        if traj:
            traj.reward = reward
        
        return EvalOutput(
            reward=reward,
            is_correct=is_valid and percentile >= 0.5,
            signals=[
                Signal(name="percentile", value=percentile),
                Signal(name="valid_submission", value=float(is_valid)),
                Signal(name="submission_provided", value=1.0),
                Signal(name="timeout", value=float(test_result.outcome == "timeout")),
            ],
        )
    
    finally:
        backend.container.stop()
```

#### 4. Prompts (`prompts/system.py`)

```python
"""MLE-Bench system prompts (ported from AMAIA)."""

SYSTEM_PROMPT = '''
You are an AUTONOMOUS ML engineering agent. Your goal is to develop an innovative 
solution that trains a model and generates predictions for a Kaggle-style competition.

<core_requirements>
Your solution MUST:
1. **Train a model** during the evaluation run
2. **Generate predictions** for the test set in `/root/data/`
3. **Write predictions to `/workspace/submission.csv`**
4. **Complete within {eval_timeout_hrs} hours** on 1× H200 GPU, 120GB RAM
5. **Match test set IDs perfectly**
</core_requirements>

<environment>
- Each tool call starts fresh; only filesystem persists
- Individual tool calls timeout after {timeout_min} minutes
- Total context window: {context_size} tokens
- Network: Enabled for pip/mamba installs
- Data location: `/root/data/` (read-only)
</environment>

<available_tools>
bash: Execute shell commands
edit: Edit existing files with search/replace
create: Create new files
submit: Submit final solution.py
</available_tools>

<tool_format>
<tool: bash>
ls /root/data
</tool>

<tool: submit>
/workspace/solution.py
</tool>
</tool_format>
'''.strip()


INSTANCE_PROMPT = '''
<task_description>
{task_description}
</task_description>

<data_info>
{data_info}
</data_info>

<instructions>
Solve the task following the guidelines above. Submit your final solution when ready.
</instructions>
'''.strip()
```

#### 5. Training Script (`train.py`)

```python
"""Train MLE-bench agent using rLLM."""

import hydra
from omegaconf import DictConfig

from mle_bench_flow import mle_bench_agent
from evaluator import mle_bench_evaluator

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig):
    # Load MLE-bench dataset
    train_dataset = DatasetRegistry.load_dataset("mle-bench", "train")
    test_dataset = DatasetRegistry.load_dataset("mle-bench", "test")
    
    if train_dataset is None:
        raise RuntimeError("MLE-bench dataset not found. Configure data path.")
    
    trainer = AgentTrainer(
        backend=config.backend,  # "tinker" or "verl"
        agent_flow=mle_bench_agent,
        evaluator=mle_bench_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

#### 6. Default Config (`configs/default.yaml`)

```yaml
# MLE-Bench training configuration

backend: verl

model:
  name: Qwen/Qwen3-8B
  lora_rank: 32

training:
  group_size: 4           # Rollouts per task for GRPO
  batch_size: 8
  epochs: 5
  learning_rate: 1e-5
  
algorithm:
  estimator: grpo
  kl_coef: 0.01

mle_bench:
  agentbox_manager_uri: "http://agentbox-manager:8080"
  max_turns: 128
  session_timeout: 360
  eval_timeout: 300
  rollout_timeout: 32400   # 9 hours
  context_size: 131072
  think: true
  message_format: xml

data:
  dataset: mle-bench
  train_split: train
  test_split: test
```

### Implementation Phases

#### Phase 1: Core Infrastructure
1. Create `cookbooks/mle_bench/` directory structure
2. Port prompts from AMAIA
3. Implement AgentBox wrapper with minimal coupling
4. Basic rollout function (single turn for testing)

#### Phase 2: Full Agent Loop
1. Complete multi-turn rollout with tool parsing
2. Implement all tools (bash, edit, create, submit)
3. Handle timeouts and error cases
4. Add conversation history management

#### Phase 3: Evaluation
1. Port `agentbox_rpc_eval` integration
2. Implement percentile scoring
3. Create `@rllm.evaluator` wrapper
4. Test with sample submissions

#### Phase 4: Training Integration
1. Create Hydra configs
2. Test with tinker backend (single machine)
3. Test with verl backend (distributed)
4. Add checkpoint/logging

#### Phase 5: Refinement
1. Add self-refinement rollout mode (multiple iterations)
2. Add important_lesson injection
3. Performance optimization
4. Documentation

### Dependencies

```toml
# cookbooks/mle_bench/pyproject.toml
[project]
name = "mle-bench-rllm"
dependencies = [
    "rllm",
    "openai",
    "hydra-core",
    "pandas",
    "numpy",
]

[project.optional-dependencies]
amaia = [
    # Include AMAIA path in PYTHONPATH for agentbox imports
]
```

### Key Design Decisions

1. **Reuse AMAIA's AgentBox**: Import directly rather than re-implementing container management. This ensures compatibility and reduces maintenance burden.

2. **Thin wrapper pattern**: `AgentBoxWrapper` provides a clean interface while delegating to AMAIA internals.

3. **Standard rLLM patterns**: Use `@rllm.rollout` and `@rllm.evaluator` decorators so MLE-bench works with existing training infrastructure.

4. **Percentile as reward**: Use percentile ranking (0-1) as the reward signal, enabling meaningful gradient updates even for submissions that don't reach top ranks.

5. **Episode structure**: Single `Trajectory` per `Episode` since MLE-bench is single-agent (unlike solver-judge which has multiple trajectories).

---

## Key Files Reference

| File | Description |
|------|-------------|
| `solver_judge_flow.py` | Agent flow defining solver + judge |
| `evaluator.py` | Reward function with per-trajectory scoring |
| `train.py` | Training script using `AgentTrainer` |
| `rllm/types.py` | Core types: `Episode`, `Trajectory`, `Step` |
| `rllm/experimental/unified_trainer.py` | Backend-agnostic trainer |
| `rllm-model-gateway/` | Transparent proxy for capturing token IDs/logprobs |
