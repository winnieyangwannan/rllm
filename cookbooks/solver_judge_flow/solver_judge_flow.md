# Understanding Agentic RL: Solver-Judge Flow

## Overview

Agentic RL trains AI agents by:
1. **Running the agent** (rollout) to produce actions
2. **Evaluating those actions** with a reward function
3. **Using RL algorithms** (like GRPO) to optimize the policy

## The Three Components

### 1. The Agent Flow (`solver_judge_flow.py`)

```python
@rllm.rollout(name="solver-judge")
def solver_judge_flow(task: Task, config: AgentConfig) -> Episode:
```

This defines a **multi-agent system**:
- **Solver** (×N): Generates candidate solutions in parallel using `OpenAI` client
- **Judge**: Evaluates and picks the best solution

Key insight: The agent uses a **plain OpenAI client** — no special RL code. The `base_url` points to the **model gateway** during training, which transparently captures token IDs and logprobs for RL optimization.

The output is an `Episode` containing multiple `Trajectory` objects (one per solver + one for judge), each with `Step`s recording the LLM calls.

### 2. The Evaluator (`evaluator.py`)

```python
@rllm.evaluator
def solver_judge_countdown_evaluator(task: dict, episode: Episode) -> EvalOutput:
```

The evaluator assigns **per-trajectory rewards**:
- Each solver trajectory: `reward = 1.0` if correct, `0.0` otherwise
- Judge trajectory: `reward = 1.0` if it picked a correct answer

This enables **independent credit assignment** — GRPO computes advantages separately for "solver" vs "judge" trajectory groups.

### 3. The Trainer (`train.py`)

```python
trainer = AgentTrainer(
    backend="tinker",           # Single-machine RL backend
    agent_flow=solver_judge_flow,
    evaluator=solver_judge_countdown_evaluator,
    config=config,
    train_dataset=train_dataset,
)
trainer.train()
```

The `AgentTrainer` orchestrates:
1. Running rollouts in parallel
2. Computing rewards via the evaluator
3. Calculating advantages (GRPO groups trajectories by name)
4. Updating the policy

## Data Flow

```
Task → Agent (solver + judge) → Episode → Evaluator → Rewards → GRPO → Policy Update
           ↑                                                              ↓
           └──────────────────── Updated Model ←─────────────────────────┘
```

## Why This Works

1. **Zero code changes for RL**: The agent uses standard `OpenAI` client — the model gateway intercepts calls and captures training data transparently.

2. **Multi-trajectory rewards**: Each trajectory gets its own reward, so the solver and judge learn independently.

3. **Standard primitives**: `Episode`, `Trajectory`, `Step` in `rllm/types.py` are the canonical data model that flows through the entire pipeline.

4. **Backend agnostic**: Same code works with `tinker` (single-machine) or `verl` (distributed GPU).

---

## Verl Backend Integration

When you use `verl` as backend, here's how the integration works:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent Code (solver_judge_flow)                                 │
│       ↓                                                         │
│  OpenAI Client (base_url → Model Gateway)                       │
│       ↓                                                         │
│  Model Gateway (transparent proxy)                              │
│   - Injects logprobs=True, return_token_ids=True                │
│   - Captures traces (prompt_ids, response_ids, logprobs)        │
│       ↓                                                         │
│  vLLM Servers (managed by Verl's AsyncLLMServerManager)         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Differences: Tinker vs Verl

| Aspect | Tinker | Verl |
|--------|--------|------|
| **Scale** | Single machine | Distributed GPU cluster (Ray) |
| **Inference** | Local model | vLLM servers with weight sync |
| **Gateway mode** | `thread` | `process` (separate process for isolation) |
| **Weight updates** | In-memory | `checkpoint_manager.update_weights()` syncs to vLLM workers |

### Data Flow (Detailed)

1. **Rollout Phase**: Agent runs via `AgentSdkEngine` → Model Gateway intercepts OpenAI calls → vLLM generates with logprobs → Traces stored in SQLite

2. **Transform Phase**: Episodes → `TrajectoryGroup` (grouped by task_id + trajectory_name like `"task1:solver"`, `"task1:judge"`)

3. **Rejection Sampling**: Drop tasks where all rollouts are correct OR all wrong — keep only "partially solved" for contrastive learning

4. **Advantage Computation**: Per-role advantage (solver/judge can use different estimators like GRPO vs RLOO)

5. **PPO Update**: 
   ```python
   old_log_prob = actor_rollout_wg.compute_log_prob(batch)
   batch = compute_advantage(batch, ...)
   actor_output = actor_rollout_wg.update_actor(batch)
   checkpoint_manager.update_weights()  # Sync to vLLM servers
   ```

### Model Gateway's Role

From `rllm-model-gateway/src/rllm_model_gateway/middleware.py`:

```python
def _mutate(self, payload, session_id):
    payload["logprobs"] = True           # Force capture
    payload["return_token_ids"] = True   # vLLM returns token IDs
```

The gateway extracts `prompt_token_ids`, `completion_token_ids`, and per-token `logprobs` — exactly what GRPO needs for policy gradients.

### Using Verl Backend

```python
trainer = AgentTrainer(
    backend="verl",              # Switch from tinker to verl
    agent_flow=solver_judge_flow,
    evaluator=solver_judge_countdown_evaluator,
    config=config,
    train_dataset=train_dataset,
)
trainer.train()  # Launches Ray, starts vLLM servers, runs distributed training
```

---

## Comparison: rLLM vs AMAIA (`/home/winnieyangwn/amaia-collab`)

Both codebases implement agentic RL for LLM training. Here are the parallel components:

### Data Types

| Concept | rLLM (`rllm/types.py`) | AMAIA (`apps/rl/`) |
|---------|------------------------|---------------------|
| **Single LLM call** | `Step` (input, output, action, reward) | `Transition` (action, observation, rewards, terminal) |
| **Agent trajectory** | `Trajectory` (list of Steps) | `Trajectory` (list of Transitions) |
| **Full rollout** | `Episode` (list of Trajectories) | `RolloutInfo` wrapping `Trajectory` |
| **Training batch** | `TrajectoryGroup` | `WorkerBatch` (list of RolloutInfo) |

### Key Difference: Token Granularity
- **rLLM**: Operates at **Step level** (per LLM call rewards), with token IDs captured by Model Gateway
- **AMAIA**: Operates at **token level** natively — `Transition` includes per-token rewards: `rewards: list[float]`

### Agent/Environment Definition

| Component | rLLM | AMAIA |
|-----------|------|-------|
| **Agent definition** | `@rllm.rollout` decorator | `Env` class with `start()` and `step()` methods |
| **Reward function** | `@rllm.evaluator` decorator | `RewardFn` class implementing `__call__(Transition) → list[float]` |
| **Environment** | Not explicit (agent handles interaction) | Explicit `Env` abstraction with `State`, `Transition` |

### rLLM Pattern:
```python
@rllm.rollout(name="solver-judge")
def solver_judge_flow(task: Task, config: AgentConfig) -> Episode:
    # Uses OpenAI client directly
    response = client.chat.completions.create(...)
    return Episode(trajectories=[...])

@rllm.evaluator
def solver_judge_evaluator(task: dict, episode: Episode) -> EvalOutput:
    # Compute rewards per trajectory
    return EvalOutput(reward=1.0, is_correct=True)
```

### AMAIA Pattern:
```python
class CodeEnv(Env):
    def start(self, episode_args: dict) -> tuple[State, Transition]:
        # Initialize environment
        return state, initial_transition
    
    def step(self, state: State, action: list[int]) -> Transition:
        # Execute action, return observation + outcomes
        return Transition(action=action, observation=obs, outcomes={"pass": True})

class PassOnlyRewardFn(RewardFn):
    def __call__(self, tr: Transition) -> list[float]:
        return [0.0] * (len(tr.action) - 1) + [2.0 * tr.outcomes["pass"] - 1.0]
```

### GRPO Implementation

| Aspect | rLLM | AMAIA |
|--------|------|-------|
| **Location** | `rllm/experimental/common/advantage.py` | `apps/rl/lib/trainer.py` |
| **Grouping** | By `task_id:trajectory_name` (e.g., `"task1:solver"`) | By prompt (all rollouts from same `start_args`) |
| **Advantage formula** | `reward - mean(group_rewards)` | `return - mean(batch_returns)` or normalized |
| **Normalization options** | Per-trajectory-group | `mean`, `mean_std`, `mean_token`, `none` |

### AMAIA GRPO Advantage:
```python
# apps/rl/lib/trainer.py
traj_returns = [sum(sum(ctx_rewards) for ctx_rewards in r.traj.rewards) for r in batch.rollouts]
mean_return = np.mean(traj_returns)
std_return = max(np.std(traj_returns), 1e-8)

# Per-context advantage
ctx["advantage"] = [(ctx_return - mean_return) / std_return]  # mean_std normalization
```

### Architecture

| Layer | rLLM | AMAIA |
|-------|------|-------|
| **Rollout execution** | `AgentFlowEngine` / `AgentSdkEngine` | `Worker` class |
| **Batch preparation** | `transform_episodes_to_trajectory_groups()` | `Trainer.get_batch()` |
| **Training loop** | `UnifiedTrainer` / `AgentTrainer` | `train.py` main loop |
| **Distributed backend** | Verl (via Ray + vLLM) | TorchDistributed + fastgen |
| **LLM inference** | Model Gateway → vLLM | `ImpGen` (importance sampling generation) |

### Key Design Differences

1. **Agent Abstraction**:
   - **rLLM**: Agents are plain Python functions with `@rllm.rollout`. Uses standard OpenAI client — **zero code changes for RL**.
   - **AMAIA**: Agents interact with explicit `Env` objects. Environment handles state transitions.

2. **Trace Capture**:
   - **rLLM**: Model Gateway transparently intercepts OpenAI API calls, captures token IDs + logprobs.
   - **AMAIA**: `ImpGen` (importance generation) handles inference + logprob capture natively.

3. **Multi-Turn/Multi-Agent**:
   - **rLLM**: Multiple `Trajectory` objects in one `Episode`, each with a `name` for independent credit assignment.
   - **AMAIA**: Single `Trajectory` per rollout, but with context switches for multi-turn. Multi-agent not explicit.

4. **Async vs Sync**:
   - **rLLM**: Supports both sync and async training (see `rllm/experimental/fully_async/`).
   - **AMAIA**: Designed for **async streaming RL** — workers and trainers operate continuously.

### Where to Find Things

| Purpose | rLLM | AMAIA |
|---------|------|-------|
| Core types | `rllm/types.py` | `apps/rl/lib/datatypes.py`, `apps/rl/envs/api.py` |
| Trainer | `rllm/experimental/unified_trainer.py` | `apps/rl/lib/trainer.py`, `apps/rl/train.py` |
| Worker/Rollout | `rllm/engine/rollout/` | `apps/rl/lib/worker.py` |
| Losses | `rllm/trainer/verl/` | `apps/rl/lib/losses.py` |
| Environments | N/A (agents handle) | `apps/rl/envs/` |
| Reward functions | `@rllm.evaluator` | `apps/rl/envs/rewards.py` |

---

