# Multi-Node MLE-Bench Evaluation Plan

## High-Level Goal

Scale the working single-node `eval.py` to run across multiple SLURM nodes using **Ray** for task distribution. This serves two purposes:

1. **Immediate:** Run 100s of MLE-bench rollouts in parallel across N nodes for evaluation
2. **Future:** Provide a Ray-based rollout generation component that plugs directly into rllm's RL training loop (which already uses Ray for distributed training via Verl backend)

## Why Ray (Not torch.distributed)

| Consideration | torch.distributed (current `distributed.py`) | Ray (proposed) |
|---|---|---|
| rllm training integration | Would need rewrite — rllm uses Ray | Direct reuse — same runtime |
| Agent rollouts | Overkill (designed for gradient sync) | Perfect fit (task parallelism) |
| Fault tolerance | One rank dies = all die | Failed task retried automatically |
| Scheduling | Manual shard calculation | Automatic load balancing |
| Existing code | `distributed.py` has buggy gather_results | `ray_init_utils.py` already battle-tested |

**Decision:** Delete `distributed.py` and `eval_distributed.py`. Replace with Ray-based `eval_ray.py`.

---

## Architecture Overview

```
User runs: python launch.py --config configs/gpt5.yaml --name exp_001 --task mlsp-2013-birds --nodes 3 --samples 192

launch.py:
  1. Creates dump directory with code copy
  2. Generates SLURM sbatch script
  3. Submits job

SLURM allocates 3 nodes, runs sbatch script:
  Node 0: ray start --head
  Node 1: ray start --address=<head>
  Node 2: ray start --address=<head>
  Node 0: python eval_ray.py --config ... --samples 192

eval_ray.py:
  1. ray.init(address="auto")  # connect to cluster
  2. Load config + task data
  3. Submit 192 Ray remote tasks → Ray schedules across 3 nodes
  4. Collect results as they complete, save trajectories
  5. Print summary on head node
```

---

## Full Flow: Multi-Node, Multi-Task Parallel Evaluation

This section provides a detailed walkthrough of how multi-node, multi-task parallelism works end-to-end.

### 1. Launch via SLURM (`launch.py`)

```bash
python launch.py --config configs/gpt5.yaml --nodes 2 --samples 64
```

This generates an sbatch script that:
1. Allocates 2 nodes (192 CPUs each)
2. Starts Ray head on node 0
3. Starts Ray worker on node 1
4. Runs `eval_ray.py` on the head node

### 2. Task Loading (`eval_ray.py` → `eval.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ eval_ray.py main()                                               │
│                                                                  │
│  1. Parse CLI args: --tasks "task1,task2,task3" --samples 64    │
│  2. Load config: cfg = load_config("configs/gpt5.yaml")         │
│  3. task_ids = ["task1", "task2", "task3"]                      │
│                                                                  │
│  4. ray.init() → connects to Ray cluster (2 nodes)              │
│                                                                  │
│  5. Calls: run_all_tasks_parallel(task_ids, cfg, num_samples)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ run_all_tasks_parallel()                                         │
│                                                                  │
│  # SINGLE FILE READ - loads all tasks at once                   │
│  task_data_map = load_tasks_from_jsonl(                         │
│      task_ids=["task1", "task2", "task3"],                      │
│      task_path="/path/to/tasks.jsonl"                           │
│  )                                                               │
│                                                                  │
│  # Returns:                                                      │
│  # {                                                             │
│  #   "task1": {"instance_id": "task1", "task_description": ...},│
│  #   "task2": {"instance_id": "task2", "task_description": ...},│
│  #   "task3": {"instance_id": "task3", "task_description": ...},│
│  # }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Build Work Queue (All Task×Sample Pairs)

```python
# 3 tasks × 64 samples = 192 total rollouts
work_queue = [
    ("task1", 0), ("task1", 1), ..., ("task1", 63),   # 64 items
    ("task2", 0), ("task2", 1), ..., ("task2", 63),   # 64 items
    ("task3", 0), ("task3", 1), ..., ("task3", 63),   # 64 items
]
# Total: 192 (task_id, sample_idx) pairs
```

### 4. Submit to Ray with Sliding Window Throttling

The sliding window pattern keeps nodes busy without overwhelming the cluster:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Sliding Window Submission (max_concurrent=64 per node = 128 total)     │
│                                                                         │
│  pending = []                                                           │
│  next_work = 0                                                          │
│                                                                         │
│  while completed < 192:                                                 │
│      # Submit up to max_concurrent tasks                                │
│      while len(pending) < 128 and next_work < 192:                     │
│          task_id, sample_idx = work_queue[next_work]                   │
│          task_data = task_data_map[task_id]  # Already loaded!         │
│                                                                         │
│          future = run_rollout_task.remote(task_data, sample_idx, cfg)  │
│          pending.append((task_id, sample_idx, future))                 │
│          next_work += 1                                                │
│                                                                         │
│      # Wait for any task to complete                                   │
│      done, _ = ray.wait([f for _, _, f in pending], num_returns=1)     │
│                                                                         │
│      # Process completed, save result                                  │
│      # Loop continues...                                               │
└────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** As soon as 1 rollout finishes, the slot is immediately refilled from the queue. No idle time as long as there's work remaining.

### 5. Ray Distributes Tasks Across Nodes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          RAY SCHEDULER                                    │
│                                                                           │
│  128 concurrent tasks → Ray auto-distributes across 2 nodes              │
│                                                                           │
│  ┌─────────────────────────┐    ┌─────────────────────────┐              │
│  │      NODE 0 (Head)      │    │      NODE 1 (Worker)    │              │
│  │   192 CPUs, 8 GPUs      │    │   192 CPUs, 8 GPUs      │              │
│  │                         │    │                         │              │
│  │  @ray.remote(num_cpus=1)│    │  @ray.remote(num_cpus=1)│              │
│  │                         │    │                         │              │
│  │  Running ~64 tasks:     │    │  Running ~64 tasks:     │              │
│  │  • (task1, 0)           │    │  • (task1, 32)          │              │
│  │  • (task2, 5)           │    │  • (task2, 17)          │              │
│  │  • (task3, 12)          │    │  • (task3, 41)          │              │
│  │  • (task1, 8)           │    │  • (task2, 63)          │              │
│  │  • ...                  │    │  • ...                  │              │
│  │                         │    │                         │              │
│  │  (mixed task1/2/3!)     │    │  (mixed task1/2/3!)     │              │
│  └─────────────────────────┘    └─────────────────────────┘              │
│                                                                           │
│  KEY: No node affinity → Ray freely mixes tasks across nodes             │
│       Easy samples finish first → workers pick up remaining hard ones    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6. Inside Each Ray Task

```python
@ray.remote(num_cpus=1)
def run_rollout_task(task_data: dict, sample_idx: int, cfg_dict: dict) -> dict:
    """Runs on ANY node Ray schedules it to."""
    
    # 1. Reconstruct config (OmegaConf not picklable)
    cfg = OmegaConf.create(cfg_dict)
    
    # 2. Run the actual rollout (same as single-node eval.py)
    result = run_single_rollout(task_data, sample_idx, cfg)
    #   └── Creates sandbox
    #   └── Builds prompts from task_data["task_description"]
    #   └── Calls _run_agent_loop() (LLM API calls)
    #   └── Runs MLEEvaluator
    #   └── Returns EvalResult
    
    # 3. Convert to dict for Ray serialization
    return dataclasses.asdict(result)
```

### 7. Results Flow Back to Head Node

```
┌────────────────────────────────────────────────────────────────────────┐
│ Head Node (run_all_tasks_parallel continues...)                         │
│                                                                         │
│  while completed < 192:                                                 │
│      done, _ = ray.wait(pending_futures, num_returns=1)                │
│                                                                         │
│      for task_id, sample_idx, future in pending:                       │
│          if future in done:                                            │
│              result_dict = ray.get(future)  # ← Result from worker     │
│              result = EvalResult(**result_dict)                        │
│                                                                         │
│              # Track per-task progress                                  │
│              all_results[task_id].append(result)                       │
│              task_completed[task_id] += 1                              │
│                                                                         │
│              # Save incrementally to JSONL                             │
│              with open(f"{task_id}.jsonl", "a") as f:                  │
│                  f.write(json.dumps(episode_dict) + "\n")              │
│                                                                         │
│              # Progress output:                                         │
│              # [45/192] ✓ task2[15/64] sample=14 percentile=0.85       │
│                                                                         │
│      # Submit new tasks to fill the window                             │
│      ...                                                               │
└────────────────────────────────────────────────────────────────────────┘
```

### 8. Output Files

```
output_dir/
├── task1.jsonl   # 64 lines (one per sample)
├── task2.jsonl   # 64 lines
└── task3.jsonl   # 64 lines
```

### 9. Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SLURM Job Submission                              │
│  python launch.py --nodes 2 --samples 64 --tasks "t1,t2,t3"             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Ray Cluster Startup (via srun --overlap)                                │
│    Node 0: ray start --head                                              │
│    Node 1: ray start --address=<head>                                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  eval_ray.py on Head Node                                                │
│    1. load_tasks_from_jsonl() → {t1: {...}, t2: {...}, t3: {...}}       │
│    2. Build work_queue: [(t1,0), (t1,1), ..., (t3,63)]  (192 items)     │
│    3. Submit to Ray with sliding window (max 128 concurrent)             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │                                                   │
           ▼                                                   ▼
┌──────────────────────────────┐            ┌──────────────────────────────┐
│  Node 0: ~64 Ray tasks       │            │  Node 1: ~64 Ray tasks       │
│  run_rollout_task.remote()   │            │  run_rollout_task.remote()   │
│                              │            │                              │
│  • Mixed t1/t2/t3 samples    │            │  • Mixed t1/t2/t3 samples    │
│  • Each: sandbox → agent →   │            │  • Each: sandbox → agent →   │
│          eval → EvalResult   │            │          eval → EvalResult   │
└──────────────────────────────┘            └──────────────────────────────┘
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Results collected on Head Node                                          │
│    • ray.wait() gets completed results                                  │
│    • Saves to t1.jsonl, t2.jsonl, t3.jsonl incrementally                │
│    • print_summary() at end                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10. Load Balancing Example

When tasks have different difficulty levels:

```
Example: 4 tasks × 64 samples on 2 nodes (max_concurrent=64 per node = 128)

Total rollouts: 256
Concurrent capacity: 128
Queued at start: 128

Time →
        0        T1       T2       T3       ...      T_end
        ├────────┼────────┼────────┼────────────────────┤
        │        │        │        │                    │
Running │==128===│==128===│==128===│======...=====128===│
        │        │        │        │                    │
Queued  │  128   │  127   │  126   │      ...      0    │
        │        │        │        │                    │
Done    │   0    │   1    │   2    │      ...     256   │


If task1 is easy (5 min/rollout) and task4 is hard (30 min/rollout):

  Early phase:
    - task1 rollouts finish quickly → freed slots pick up queued work
    - task4 rollouts still running

  Mid phase:
    - All 64 task1 rollouts DONE
    - task2, task3 progressing
    - task4 still grinding (but getting more slots as others finish)

  End phase:
    - Only hard task4 rollouts remain
    - Both nodes working on task4
    - No idle nodes!

Without global load balancing (node affinity):
  - Node 0 finishes task1 in 5 hours, sits idle
  - Node 1 still grinding task4 for 30 hours
  - 50% wasted capacity!
```

**Key benefits of this design:**
1. **Single file read** — `load_tasks_from_jsonl()` loads all tasks in one pass
2. **Global work pool** — All `(task, sample)` pairs in one queue, Ray auto-balances
3. **No idle nodes** — If task1 is easy, freed workers immediately pick up task2/task3 samples
4. **Sliding window** — Prevents overwhelming the cluster while keeping it busy

---

## Critical Refactor: Separate Rollout from Evaluation in `eval.py`

### Problem

The current `run_single_rollout()` (eval.py:132-311) is a **monolith** that conflates three concerns:

1. **Client creation** (lines 214-219): `openai.AzureOpenAI` hardcoded inside
2. **Agent rollout** (lines 167-238): sandbox setup, prompt building, agent loop
3. **Evaluation** (lines 253-277): `MLEEvaluator` called inline, scores computed

For training (Stage 4), rllm requires:
- Client **injected** by rllm's SDK proxy (captures token IDs + logprobs for policy gradient)
- Evaluator called **separately** via `@rllm.evaluator` protocol
- Rollout returns `Trajectory`, not `EvalResult` with scores baked in

If we leave the monolith as-is, Stage 4 would require **duplicating** all sandbox/prompt/agent logic in a new wrapper. That's not modular.

### Solution: Decompose into 3 functions

```python
# eval.py — refactored structure

def create_client(cfg, base_url: str | None = None):
    """Create LLM client from config.

    Args:
        cfg: Config with model settings (azure_endpoint, api_key, etc.)
        base_url: Override URL for training (rllm gateway session URL).
                  If provided, creates OpenAI client pointing to gateway.
                  If None, creates AzureOpenAI client for direct eval.

    This pattern matches rllm convention: agents create their own client
    using a base_url, which can be the gateway's session URL for training.
    """
    import openai
    if base_url:
        # Training mode: use gateway session URL (captures token IDs + logprobs)
        return openai.OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        # Eval mode: direct Azure OpenAI
        return openai.AzureOpenAI(
            azure_endpoint=cfg.model.azure_endpoint,
            api_key=cfg.model.api_key,
            api_version=cfg.model.api_version,
        )

def run_agent(task_data, sample_idx, cfg, base_url: str | None = None):
    """Pure rollout: sandbox → agent loop → result.

    No evaluation, no score computation.
    Client URL is injectable (defaults to AzureOpenAI for eval).

    Args:
        task_data: Task dict with instance_id, description, etc.
        sample_idx: Sample index for this rollout
        cfg: Config with model/data settings
        base_url: Override URL for training. Matches rllm convention where
                  agents create clients internally using config.base_url.

    This is THE reusable core for both eval and training.

    Returns:
        RolloutOutput(steps, messages, pred_solution, sandbox)
    """
    client = create_client(cfg, base_url)

    # sandbox setup (unchanged from current eval.py)
    # prompt building (unchanged from current eval.py)
    # agent loop — calls _run_agent_loop from mle_agent.agent:
    #   steps, messages, pred_solution, rollout_metrics = _run_agent_loop(
    #       client=client,  # <-- client injected here
    #       model=cfg.model.name,
    #       messages=messages,
    #       sandbox=sandbox,
    #       ...other params...
    #   )

    return RolloutOutput(steps, messages, pred_solution, sandbox, task_id, sample_idx, duration)

def evaluate_rollout(task_data, pred_solution, sandbox, cfg) -> EvalOutput:
    """Run MLEEvaluator on a completed rollout.

    Separated so training uses @rllm.evaluator instead.

    Returns:
        EvalOutput from MLEEvaluator (already proper rllm type with:
        - reward: float (percentile)
        - is_correct: bool
        - signals: list[Signal] with "percentile", "raw_score", etc.)
    """
    from mle_agent.evaluator import MLEEvaluator

    # MLEEvaluator expects Episode-like object with artifacts
    class _EvalEpisode:
        def __init__(self, pred_solution, sandbox):
            self.artifacts = {"_sandbox": sandbox, "pred_solution": pred_solution}

    evaluator = MLEEvaluator(
        mle_bench_data_dir=cfg.data.mle_bench_data_dir,
        eval_timeout=cfg.agent.get("eval_timeout", 32400),
        submit_file=cfg.agent.submit_file,
    )
    task = {"task_id": task_data["instance_id"], "instance_id": task_data["instance_id"]}
    episode = _EvalEpisode(pred_solution, sandbox)

    # MLEEvaluator.evaluate() returns proper EvalOutput (rllm type)
    return evaluator.evaluate(task, episode)

def run_single_rollout(task_data, sample_idx, cfg):
    """Full eval pipeline: rollout + evaluate.

    100% backward compatible — same signature, same return type.
    Just now composed from run_agent + evaluate_rollout.
    """
    output = run_agent(task_data, sample_idx, cfg)
    eval_output = evaluate_rollout(task_data, output.pred_solution, output.sandbox, cfg)
    return EvalResult(
        task_id=..., percentile=eval_output.reward,
        steps=output.steps, messages=output.messages, ...
    )
```

### Why This Matters

| Scenario | What gets called |
|---|---|
| **Eval (Stage 1-3)** | `run_single_rollout()` → calls `run_agent()` + `evaluate_rollout()` internally |
| **Training (Stage 4)** | `run_agent(base_url=gateway_session_url)` directly + `@rllm.evaluator` wraps `evaluate_rollout()` |

Zero duplication. The training integration is just wiring:

```python
from rllm.experimental.eval.types import Task, AgentConfig, EvalOutput

@rllm.rollout
def mle_bench_rollout(task: Task, config: AgentConfig) -> Episode:
    # rllm convention: use config.base_url (gateway session URL)
    output = run_agent(task.data, 0, cfg, base_url=config.base_url)
    return Episode(
        id=task.data["instance_id"],
        task=task.data,
        trajectories=[Trajectory(steps=output.steps, output=output.pred_solution)],
        artifacts={"pred_solution": output.pred_solution, "_sandbox": output.sandbox},
    )

@rllm.evaluator
def mle_bench_evaluator(task: dict, episode: Episode) -> EvalOutput:
    # evaluate_rollout() returns EvalOutput from MLEEvaluator
    # (already has proper reward, is_correct, signals: list[Signal])
    return evaluate_rollout(
        task, episode.artifacts["pred_solution"],
        episode.artifacts["_sandbox"], cfg,
    )
```

---

## What We Reuse (No Changes Needed)

| Component | File | Role |
|---|---|---|
| `EvalResult` | `eval.py:56` | Result dataclass — serialized via `dataclasses.asdict()` for Ray |
| `load_config()` | `eval.py:74` | YAML config loading with defaults/inheritance |
| `load_task_from_jsonl()` | `eval.py:107` | Task data loading from JSONL |
| `save_trajectory()` | `eval.py:312` | Trajectory saving to JSON |
| `print_summary()` | `eval.py:447` | Results summary printing |
| `run_task_eval()` | `eval.py:360` | Task-level orchestration — calls `run_single_rollout` |
| `get_ray_init_settings()` | `rllm/trainer/ray_init_utils.py:39` | Ray cluster auto-detection |
| Config files | `configs/base.yaml`, `configs/gpt5.yaml` | Existing configs work as-is |
| `launch.py` | `launch.py:70` | SLURM launcher — extend with Ray cluster startup |

## What We Refactor (Backward Compatible)

| Component | Change | Why |
|---|---|---|
| `run_single_rollout()` | Decompose into `run_agent()` + `evaluate_rollout()` + composition | Training needs rollout without inline eval, injectable base_url |
| New: `create_client(cfg, base_url)` | Extract client creation with URL override | Training passes gateway session URL; eval uses None for AzureOpenAI |
| New: `run_agent(..., base_url)` | Pure rollout, no eval, URL-injectable | Reusable core — matches rllm pattern where agents create clients from base_url |
| New: `evaluate_rollout()` → `EvalOutput` | Standalone evaluation returning proper type | Training wraps with `@rllm.evaluator`; returns `MLEEvaluator.evaluate()` output directly (already proper `EvalOutput` with `list[Signal]`) |

`run_single_rollout()` keeps its exact same signature and return type — it just delegates internally.

---

## Stage 0: Refactor `eval.py` — Separate Rollout from Evaluation

### Goal
Decompose `run_single_rollout` into composable pieces. Fully backward compatible.

### Changes to `eval.py`

**Note:** `RolloutOutput` and `EvalResult` dataclasses already exist in the current eval.py (lines 62-73 and 76-99). The refactor primarily involves extracting functions and adding `base_url` parameter.

1. `RolloutOutput` dataclass (already exists — may need `rollout_metrics` field):
   ```python
   @dataclass
   class RolloutOutput:
       steps: list
       messages: list
       pred_solution: str | None
       sandbox: Any  # kept open for evaluation
       task_id: str
       sample_idx: int
       duration: float
       rollout_metrics: dict  # prompt_tokens, completion_tokens, termination_reason
   ```
2. Extract `create_client(cfg, base_url=None)` — supports URL override for training
3. Extract `run_agent(task_data, sample_idx, cfg, base_url=None)` — sandbox setup through agent loop, creates client internally using base_url (matches rllm convention)
4. Extract `evaluate_rollout(task_data, pred_solution, sandbox, cfg) -> EvalOutput` — creates `_EvalEpisode` wrapper, calls `MLEEvaluator.evaluate()`, returns its output directly (already proper `EvalOutput` with `list[Signal]`)
5. Rewrite `run_single_rollout` to compose `run_agent` + `evaluate_rollout`

### Test Plan for Stage 0

```bash
# The ONLY test needed: verify backward compatibility
# run_single_rollout must produce identical results before and after refactor
python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 1 --output-dir /tmp/test_refactor
# Compare output with a known-good run
```

---

## Stage 1: `eval_ray.py` — Ray-Based Multi-Node Eval

### Goal
Create `eval_ray.py` that dispatches rollouts as Ray remote tasks, with automatic distribution across nodes.

### Design

```python
# eval_ray.py — key structure

import ray
from rllm.trainer.ray_init_utils import get_ray_init_settings

@ray.remote(num_cpus=1)
def run_rollout_task(task_data: dict, sample_idx: int, cfg_dict: dict) -> dict:
    """Stateless Ray task wrapping existing run_single_rollout.

    Note: cfg is passed as dict (not OmegaConf) because Ray needs picklable args.
    """
    from eval import run_single_rollout, EvalResult
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_dict)
    result = run_single_rollout(task_data, sample_idx, cfg)
    return dataclasses.asdict(result)  # Ray needs picklable return

def run_task_eval_ray(task_id, cfg, num_samples, output_dir):
    """Dispatch all samples as Ray tasks, collect results."""
    task_data = load_task_from_jsonl(task_id, cfg.data.task_jsonl_dir)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Submit all tasks — Ray handles scheduling across nodes
    futures = [run_rollout_task.remote(task_data, i, cfg_dict) for i in range(num_samples)]

    # Collect as completed (ray.wait pattern for progress tracking)
    results = []
    remaining = futures
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        result_dict = ray.get(done[0])
        result = EvalResult(**result_dict)
        results.append(result)
        # Save trajectory incrementally
        if output_dir and result.steps:
            save_trajectory(result, task_data, Path(output_dir), cfg)
        # Print progress
        print(f"[{len(results)}/{num_samples}] sample={result.sample_idx} "
              f"percentile={result.percentile:.4f}")

    return results

def main():
    args = parse_args()  # same CLI args as eval.py + minor additions
    cfg = load_config(args.config)

    # Init Ray — auto-detect cluster or start local
    ray.init(**get_ray_init_settings())
    print(f"Ray cluster: {ray.cluster_resources()}")

    # Run eval for each task
    all_results = {}
    for task_id in task_ids:
        results = run_task_eval_ray(task_id, cfg, num_samples, output_dir)
        all_results[task_id] = results

    print_summary(all_results)
```

### Key Design Decisions

1. **`@ray.remote` function, not actor**: Rollouts are stateless. No need for actor state persistence. This matches the embarrassingly parallel nature of eval. **Note:** Training (Stage 4) will use Ray actors instead (via `RolloutExecutor`) for weight version tracking and connection pooling.
2. **`cfg_dict` not `cfg`**: OmegaConf objects aren't picklable. Convert to dict for Ray serialization, reconstruct inside the task.
3. **`ray.wait` loop**: Provides real-time progress instead of blocking on `ray.get(futures)`.
4. **`num_cpus=1`**: Each rollout is IO-bound (waiting on LLM API + sandbox). 1 CPU per task maximizes parallelism.
5. **No changes to `run_single_rollout`**: The existing function is called as-is inside the Ray task.

### Fallback to Single-Node

When no Ray cluster is running, `ray.init()` starts a local cluster. `eval_ray.py` works identically to `eval.py` in this case — no separate code path needed.

### Test Plan for Stage 1

```bash
# Test 1: Local mode (no cluster) — should behave like eval.py
python eval_ray.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 2

# Test 2: Verify same results as eval.py
# Run both, compare trajectory outputs (scores, percentiles should match)
python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 2 --output-dir /tmp/eval_single
python eval_ray.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 2 --output-dir /tmp/eval_ray
# Compare: both should produce 2 trajectory files with same structure

# Test 3: Check Ray dashboard shows tasks
# While eval_ray.py is running, open http://localhost:8265
# Verify tasks appear in the Ray dashboard
```

---

## Stage 2: Update `launch.py` — Ray Cluster in SLURM

### Goal
Extend `generate_sbatch_script()` to start a Ray cluster across SLURM-allocated nodes before running `eval_ray.py`.

### Design

Add a `--ray` flag to `launch.py`. When enabled, the generated sbatch script:

1. Starts Ray head on the first node
2. Starts Ray workers on remaining nodes
3. Runs `eval_ray.py` on the head node
4. Stops Ray on exit

```bash
# Key additions to the generated sbatch script:

# Get node list
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_PORT=6379

# Start Ray head on first node
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    ray start --head --port=$HEAD_PORT --num-cpus=$CPUS_PER_NODE &

sleep 10  # Wait for head to be ready

# Start Ray workers on remaining nodes
for node in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w $node \
        ray start --address=$HEAD_NODE:$HEAD_PORT --num-cpus=$CPUS_PER_NODE &
done

sleep 15  # Wait for workers to join

# Run eval on head node (Ray distributes tasks automatically)
RAY_ADDRESS=$HEAD_NODE:$HEAD_PORT python eval_ray.py \
    --config $CONFIG --samples $SAMPLES --task $TASK --output-dir $OUTPUT_DIR

# Cleanup
ray stop
```

### Changes to `launch.py`

- Add `--ray` flag (default: True when `--nodes > 1`)
- Update `generate_sbatch_script()` to include Ray cluster startup/teardown
- Change eval script from `eval.py` to `eval_ray.py` when `--ray` is set
- Remove `--distributed` flag and related torch.distributed logic (replaced by `--ray`)

### Test Plan for Stage 2

```bash
# Test 1: Dry run — inspect generated sbatch script
python launch.py --config configs/gpt5.yaml --name test_ray --task mlsp-2013-birds \
    --nodes 2 --samples 32 --dry-run
# Verify: sbatch script includes ray start/stop commands

# Test 2: Single-node launch (should still work, Ray starts locally)
python launch.py --config configs/gpt5.yaml --name test_single --task mlsp-2013-birds \
    --nodes 1 --samples 4 --dry-run
# Verify: sbatch script uses eval_ray.py without multi-node Ray setup

# Test 3: Submit real multi-node job
python launch.py --config configs/gpt5.yaml --name test_multi --task mlsp-2013-birds \
    --nodes 3 --samples 48
# Monitor: tail -f <dump_dir>/logs/<job_id>.out
# Verify: output_dir has 48 trajectory files
# Verify: logs show tasks running on all 3 nodes
```

---

## Stage 3: Config & Cleanup

### Goal
Clean up deprecated files. Config already set up in earlier stages.

### Changes

1. **`configs/base.yaml`** — already configured with:
```yaml
# SLURM settings (used by launch.py)
slurm:
  qos: h200_coding_shared
  account: aira_ws2
  time: "168:00:00"
  nodes: 1
  gpus_per_node: 1
  cpus_per_node: 192

# Ray settings (used by eval_ray.py)
ray:
  max_concurrent_tasks: 64   # Max parallel rollouts per node
```

2. **Delete deprecated files:**
   - `distributed.py` — torch.distributed approach, replaced by Ray

3. **Update `eval.py`** — add note in docstring pointing to `eval_ray.py` for multi-node

### Test Plan for Stage 3

```bash
# Test 1: Verify eval.py still works standalone (regression)
python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 1

# Test 2: Verify eval_ray.py reads distributed config
python eval_ray.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 1
```

---

## Stage 4 (Future): RL Training Integration

### Goal
Plug the multi-node Ray-based rollout into rllm's RL training loop so we can train MLE-bench agents with GRPO/REINFORCE/RLOO.

### Which Training Path to Use?

| Criteria | Path A: UnifiedTrainer | Path B: FullyAsyncTaskRunner |
|----------|----------------------|------------------------------|
| **Best for** | Quick prototyping, smaller models | Production, large models |
| **Concurrency** | Sync batches (N rollouts → train → repeat) | True async (rollout & train in parallel) |
| **GPU util** | Lower (GPUs idle during rollout) | Higher (continuous training) |
| **Staleness** | None (on-policy) | Configurable (0-10% typical) |
| **Complexity** | Simpler | More complex (weight sync, staleness mgmt) |
| **LLM client** | OpenAI-compatible (gateway) | `RolloutClient` (direct SGLang) |
| **Agent code** | Existing `_run_agent_loop()` works | Needs async `_run_agent_loop_async()` |

**Recommendation for MLE-bench:**
- **Start with Path A** for correctness testing — simpler, uses existing sync agent code
- **Graduate to Path B** for scale — requires async agent loop but gives better GPU utilization

### rllm Training Architecture (How It Works Today)

rllm offers two training paths. Both use Ray:

**Path A: UnifiedTrainer (sync, on-policy)**
```
@rllm.rollout function
  → UnifiedWorkflowEngine.execute_tasks()     # parallel rollout generation
  → Episodes (list[Episode])
  → transform_episodes_to_trajectory_groups()  # group n rollouts per task
  → backend.transform_to_backend_batch()       # → Verl DataProto
  → backend.process_backend_batch()            # compute logprobs, critic values
  → backend.compute_advantages()               # GRPO/GAE advantage estimation
  → backend.update_policy()                    # PPO/REINFORCE gradient step
```
See: `rllm/experimental/unified_trainer.py:363` (`_fit_on_policy`)

**Path B: FullyAsyncTaskRunner (async, distributed)**
```
create_task_runner_with_rollout_fn(rollout_fn)  # factory wraps as Ray actor
  → RolloutExecutor.generate_trajectory()       # async rollout dispatch
  → TrajectoryGroup enqueued to MessageQueue    # async buffer
  → FullyAsyncTrainer.fit() consumes batches    # trains while generating
  → ParameterSynchronizer broadcasts weights    # updates rollout policy
```
See: `rllm/experimental/fully_async/runner.py:172`

### Required Function Signatures

rllm expects specific signatures depending on the training path:

**For @rllm.rollout decorator (Path A):**
```python
# File: rllm/experimental/eval/rollout_decorator.py
# Protocol: rllm/experimental/eval/types.py

from rllm.experimental.eval.types import Task, AgentConfig

@rllm.rollout
def solve(task: Task, config: AgentConfig) -> Episode | str | list[Trajectory] | dict:
    """
    Args:
        task: Task wrapper with task.data dict containing task_id, task_description, etc.
        config: AgentConfig with base_url (gateway session URL), model, session_uid, metadata
    Returns:
        Episode (or simpler types that get auto-coerced to Episode)
    
    IMPORTANT: Use config.base_url to create OpenAI client — this routes through
    rllm's gateway which captures token IDs + logprobs for policy gradient.
    """
```

**For FullyAsyncTaskRunner (Path B):**
```python
# File: rllm/experimental/fully_async/rollout_executor.py:306

async def rollout_fn(client: RolloutClient, tokenizer, **kwargs) -> Trajectory:
    """
    Args:
        client: LiteLLM-compatible client that captures token IDs + logprobs
        tokenizer: For encoding/decoding (needed for training, not eval)
        **kwargs: Task data fields (instance_id, task_description, etc.)
    Returns:
        Trajectory with steps, output, reward
    """
```

### How Our Eval Code Maps to These Signatures

Our current `run_single_rollout(task_data, sample_idx, cfg) -> EvalResult` needs adaptation:

**Gap 1: Client injection via base_url** — Training needs rllm's gateway session URL (captures token IDs, logprobs for policy gradient). Our eval uses `openai.AzureOpenAI` directly. Fix: Accept `base_url` parameter in `run_agent()`, create client internally. Matches rllm convention where agents create clients from `config.base_url`.

**Gap 2: Return type** — Training expects `Episode` (rllm type). Our eval returns `EvalResult`. Fix: We already build `Trajectory` in `save_trajectory()` — extract that logic.

**Gap 3: Evaluator protocol** — Training calls evaluator separately via `@rllm.evaluator`. Our eval runs `MLEEvaluator` inline. Fix: Already have `MLEEvaluator` in `agenthub/mle_agent/mle_agent/evaluator.py` — wrap with `@rllm.evaluator`, return `EvalOutput` type.

**Gap 4: Trace enrichment** — Training requires enriched episodes with `prompt_ids`, `response_ids`, `logprobs`. rllm's `AgentFlowEngine` handles this automatically via `gateway.aget_traces()`. Fix: Use `AgentFlowEngine` or call gateway traces manually.

### Concrete Integration Plan

Because of the Stage 0 refactor, the training integration is just thin wrappers around existing functions — no logic duplication.

#### Stage 4a: `@rllm.rollout` wrapper (~20 lines of new code)

```python
# File: cookbooks/mlebench/train_integration/rollout.py

import rllm
from rllm.types import Episode, Trajectory
from rllm.experimental.eval.types import Task, AgentConfig
from eval import run_agent  # ← reuse the Stage 0 refactored function

@rllm.rollout
def mle_bench_rollout(task: Task, config: AgentConfig) -> Episode:
    """MLE-bench rollout for rllm training.

    Calls the SAME run_agent() used by eval, but with rllm's gateway
    session URL (captures token IDs + logprobs for policy gradient).

    Key rllm convention: use config.base_url to create client internally.
    The gateway handles trace capture — no manual client injection needed.
    """
    cfg = config.metadata["cfg"]
    task_data = task.data  # Unwrap Task → dict

    # Pass gateway session URL — run_agent creates client internally
    output = run_agent(task_data, 0, cfg, base_url=config.base_url)

    trajectory = Trajectory(
        name="mle_agent",
        steps=output.steps,
        output=output.pred_solution,
    )
    return Episode(
        id=f"{task_data['instance_id']}:{config.session_uid}",
        task=task_data,
        trajectories=[trajectory],
        artifacts={"pred_solution": output.pred_solution, "_sandbox": output.sandbox},
    )
```

**Note on trace enrichment:** When using `AgentFlowEngine` (the recommended path), trace capture and enrichment with `prompt_ids`, `response_ids`, `logprobs` happens automatically. The engine calls `gateway.aget_traces(uid)` and merges token-level data into the Episode. If implementing a custom training loop, you must call:
```python
traces = await gateway.aget_traces(config.session_uid)
enriched_episode = enrich_episode_with_traces(episode, traces)
```

#### Stage 4b: `@rllm.evaluator` wrapper (~15 lines of new code)

```python
# File: cookbooks/mlebench/train_integration/evaluator.py

import rllm
from rllm.types import Episode
from rllm.experimental.eval.types import EvalOutput
from eval import evaluate_rollout  # ← reuse the Stage 0 refactored function

@rllm.evaluator
def mle_bench_evaluator(task: dict, episode: Episode, cfg) -> EvalOutput:
    """MLE-bench evaluator for rllm training.

    Calls the SAME evaluate_rollout() used by eval.
    Returns EvalOutput with is_correct, reward, and signals dict.
    """
    # evaluate_rollout already returns EvalOutput (see Stage 0 refactor)
    return evaluate_rollout(
        task,
        episode.artifacts["pred_solution"],
        episode.artifacts["_sandbox"],
        cfg,
    )
```

#### Stage 4c: Training script (~15 lines of new code)

```python
# File: cookbooks/mlebench/train_integration/train.py

from rllm.experimental.unified_trainer import AgentTrainer
from rollout import mle_bench_rollout
from evaluator import mle_bench_evaluator

trainer = AgentTrainer(
    backend="verl",                    # distributed GPU training
    agent_flow=mle_bench_rollout,      # our @rllm.rollout — calls run_agent()
    evaluator=mle_bench_evaluator,     # our @rllm.evaluator — calls evaluate_rollout()
    config=config,
    train_dataset=train_dataset,       # JSONL tasks
)
trainer.train()
```

**Total new code for training integration: ~50 lines, zero duplication.**

#### Stage 4d: Async Agent Loop for FullyAsync Training (Path B)

The above Stage 4a-c works for **Path A (UnifiedTrainer)** which uses gateway + OpenAI clients. For **Path B (FullyAsyncTaskRunner)**, there's a client interface mismatch:

| Component | Path A (UnifiedTrainer) | Path B (FullyAsyncTaskRunner) |
|-----------|------------------------|------------------------------|
| Client | `openai.OpenAI(base_url=gateway_url)` | `RolloutClient` (httpx async) |
| Interface | `client.chat.completions.create()` | `await client.chat_completion()` |
| Sync/Async | Sync | Async only |
| Weight tracking | Gateway handles | `client.cur_version`, `resume_event` |

The current `_run_agent_loop()` in `mle_agent/agent.py` uses OpenAI's sync interface. For fully async training, we need an adapter.

**Option 1: RolloutClient wrapper mimicking OpenAI interface**

```python
# File: cookbooks/mlebench/train_integration/client_adapter.py

class OpenAICompatibleRolloutClient:
    """Wraps RolloutClient to mimic openai.OpenAI interface for _run_agent_loop."""
    
    def __init__(self, rollout_client: RolloutClient):
        self._client = rollout_client
        self.chat = self._ChatNamespace(rollout_client)
    
    class _ChatNamespace:
        def __init__(self, client):
            self._client = client
            self.completions = self._CompletionsNamespace(client)
        
        class _CompletionsNamespace:
            def __init__(self, client):
                self._client = client
            
            def create(self, model, messages, tools=None, **kwargs):
                """Sync wrapper around async RolloutClient."""
                import asyncio
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._acreate(model, messages, tools, **kwargs))
            
            async def _acreate(self, model, messages, tools=None, **kwargs):
                sampling_params = {"max_new_tokens": kwargs.get("max_tokens", 4096), ...}
                message, output = await self._client.chat_completion(messages, sampling_params, tools)
                # Convert to OpenAI ChatCompletion-like response
                return _make_chat_completion(message, output)
```

**Option 2: Native async agent loop (preferred for performance)**

```python
# File: mle_agent/agent_async.py

async def _run_agent_loop_async(
    client: RolloutClient,
    tokenizer,
    messages: list[dict],
    sandbox: Sandbox,
    **kwargs
) -> tuple[list[Step], list[dict], str | None, dict]:
    """Async version of _run_agent_loop for FullyAsync training.
    
    Uses RolloutClient directly (no sync wrapper overhead).
    Supports pause/resume for weight sync via client.resume_event.
    """
    while not done:
        # RolloutClient auto-handles pause/resume during weight sync
        message, output = await client.chat_completion(messages, sampling_params, tools)
        
        # Track weight version for staleness
        step.metadata["param_version"] = client.cur_version
        
        # ... rest of agent loop logic (same as sync version)
```

**Rollout function for Path B:**
```python
# File: cookbooks/mlebench/train_integration/rollout_async.py

async def mle_bench_rollout_fn(client: RolloutClient, tokenizer, **kwargs) -> Trajectory:
    """FullyAsync-compatible rollout function.
    
    Signature matches rllm/experimental/fully_async/rollout_executor.py expectations.
    """
    from mle_agent.agent_async import _run_agent_loop_async
    
    task_data = kwargs  # Task fields passed as kwargs
    sandbox = create_sandbox(task_data, cfg)
    messages = build_initial_messages(task_data, cfg)
    
    steps, final_messages, pred_solution, metrics = await _run_agent_loop_async(
        client=client,
        tokenizer=tokenizer,
        messages=messages,
        sandbox=sandbox,
        **agent_kwargs
    )
    
    trajectory = Trajectory(
        steps=steps,
        output=pred_solution,
        metadata={"param_version": client.cur_version},
    )
    return trajectory
```

**Training script for Path B:**
```python
# File: cookbooks/mlebench/train_integration/train_async.py

from rllm.experimental.fully_async.runner import create_task_runner_with_rollout_fn
from rollout_async import mle_bench_rollout_fn

# Create configured runner with our rollout function
ConfiguredRunner = create_task_runner_with_rollout_fn(
    rollout_fn=mle_bench_rollout_fn,
    val_rollout_fn=mle_bench_rollout_fn,  # Optional validation
)

# Launch training
runner = ConfiguredRunner.remote()
ray.get(runner.run.remote(config))
```

**Recommendation:** Implement Option 2 (native async) for production — it avoids sync/async bridging overhead and properly supports weight sync pause/resume. Option 1 is a quick adapter for testing.

#### Stage 4 Prerequisites: GatewayManager Setup

Training requires the `rllm-model-gateway` to be running. The gateway intercepts LLM calls and captures token IDs + logprobs needed for policy gradient computation.

```python
# In training script setup:
from rllm.experimental.engine.gateway_manager import GatewayManager

gateway = GatewayManager(port=9090, ...)
await gateway.start(rollout_engine)  # TinkerEngine or VerlEngine

# Gateway creates session URLs for each rollout:
# http://gateway:9090/session/{session_id}/v1
# These URLs are passed as config.base_url to the @rllm.rollout function
```

When using `UnifiedTrainer` or `AgentFlowEngine`, gateway setup is handled automatically. Only manual setup is needed for custom training loops.

| Component | Role |
|---|---|
| `GatewayManager` | Manages gateway lifecycle, creates session URLs |
| `gateway.acreate_session(uid)` | Creates session for trace correlation |
| `gateway.get_session_url(uid)` | Returns URL like `http://gateway:9090/session/{id}/v1` |
| `gateway.aget_traces(uid)` | Retrieves captured traces (token IDs, logprobs) |

### How Multi-Node Rollout Fits In

The Ray-based `eval_ray.py` from Stage 1-2 establishes the pattern. For training:

```
Stage 1-2 (eval):
  eval_ray.py → @ray.remote run_rollout_task() → EvalResult

Stage 4 (training):
  UnifiedTrainer → VerlBackend.generate_episodes()
    → UnifiedWorkflowEngine (uses Ray internally via Verl's RayWorkerGroup)
    → @rllm.rollout mle_bench_rollout()  ← same core logic as run_single_rollout
    → Episode → TrajectoryGroup → DataProto → gradient update
```

The multi-node Ray cluster setup from Stage 2 (SLURM launch script) is reused directly — Verl's training already expects a Ray cluster.

### Key rllm Files for Training Integration

| File | Role | What we interact with |
|---|---|---|
| `rllm/experimental/eval/rollout_decorator.py` | `@rllm.rollout` decorator | Wraps our rollout function |
| `rllm/experimental/eval/types.py` | `Task`, `AgentConfig`, `EvalOutput`, `AgentFlow`, `Evaluator` protocols | Our functions use these types |
| `rllm/experimental/engine/gateway_manager.py` | `GatewayManager` for trace capture | Creates session URLs, retrieves traces |
| `rllm/experimental/engine/agent_flow_engine.py` | `AgentFlowEngine` for rollout orchestration | Handles trace enrichment automatically |
| `rllm/experimental/unified_trainer.py` | `UnifiedTrainer` class | Main training entry point |
| `rllm/experimental/verl/verl_backend.py` | Verl backend | Handles distributed training |
| `rllm/experimental/fully_async/runner.py` | `create_task_runner_with_rollout_fn()` | Alternative async training path |
| `rllm/experimental/fully_async/rollout_executor.py` | `RolloutExecutor` Ray actor | Manages async rollout dispatch |
| `rllm/experimental/fully_async/message_queue.py` | `MessageQueue` Ray actor | Async buffer between rollout & training |
| `rllm/experimental/sync_coordinator.py` | `SyncCoordinator` | Weight sync + staleness management |
| `rllm/types.py` | `Step`, `Trajectory`, `Episode` | Core data types |
| `rllm/agents/agent.py` | Extended types with `prompt_ids`, `logprobs`, `advantage` | Training-specific step fields |

### Design Decisions That Enable Stage 4

Choices in Stage 1-3 that specifically support training integration:

1. **Ray (not torch.distributed)**: Verl training already runs on Ray. Same cluster, same runtime.
2. **`run_agent(base_url=...)` pattern**: Matches rllm convention where agents create clients internally using `config.base_url`. For eval, `base_url=None` uses AzureOpenAI; for training, `base_url=gateway_session_url` routes through the gateway. **Note:** This pattern works for Path A (UnifiedTrainer). Path B (FullyAsync) uses `RolloutClient` directly — see Stage 4d.
3. **EvalResult → Episode/Trajectory conversion exists**: `save_trajectory()` already does this. Extract for reuse.
4. **Config as dict**: Both Ray serialization and rllm's `AgentConfig.metadata` use dicts.
5. **Evaluator returns `EvalOutput`**: `MLEEvaluator.evaluate()` already returns proper `EvalOutput` type with `reward`, `is_correct`, `signals: list[Signal]`. Works for both eval (convert to `EvalResult` fields) and training (pass through to rllm).
6. **Stateless eval vs stateful training**: Eval uses `@ray.remote` functions (stateless, embarrassingly parallel). Training uses `RolloutExecutor` actors (stateful, weight version tracking). The core `run_agent()` function works in both contexts.
7. **Modular agent loop**: `_run_agent_loop()` is a separate function in `mle_agent/agent.py`. For fully async training (Path B), we add `_run_agent_loop_async()` that uses `RolloutClient` directly instead of OpenAI client — same logic, different I/O layer.

### What Changes Between Eval and Training

| Aspect | Eval (Stage 1-3) | Training (Stage 4) |
|---|---|---|
| LLM Client | `openai.AzureOpenAI` via `base_url=None` | `openai.OpenAI` via `base_url=gateway_session_url` (captures token IDs, logprobs) |
| Task type | Raw dict | `Task` wrapper (access via `task.data`) |
| Config type | OmegaConf `cfg` | `AgentConfig` with `base_url`, `session_uid`, `metadata` |
| Return type | `EvalResult` (dataclass) | `Episode` (rllm type) |
| Evaluator return | `EvalOutput` (converted to `EvalResult` fields) | `EvalOutput` (used directly by rllm) |
| Evaluator call | Inline in `run_single_rollout` | Separate via `@rllm.evaluator` |
| Trace enrichment | Not needed | Gateway captures traces → `AgentFlowEngine` enriches Episode |
| Ray pattern | Stateless `@ray.remote` functions | Stateful `RolloutExecutor` actors (weight versioning) |
| Reward usage | Logged to file | Fed into advantage estimation → gradient update |
| Weight updates | None (frozen model) | After each batch via Verl backend |
| Orchestration | `eval_ray.py` dispatches tasks | `UnifiedTrainer` or `FullyAsyncTaskRunner` orchestrates |

This stage is NOT implemented in Stages 1-3 — documenting the full path so design decisions support it and so a future agent can implement it.

---

## File Summary

### Stage 0: Refactor (backward compatible)
| File | Change |
|---|---|
| `eval.py` | Decompose `run_single_rollout` → `create_client` + `run_agent` + `evaluate_rollout` + composition. Same external behavior. |

### Stage 1: New Files
| File | Purpose |
|---|---|
| `eval_ray.py` | Multi-node eval entry point using Ray |

### Stage 2: Modified Files
| File | Change |
|---|---|
| `launch.py` | Add `--ray` flag, Ray cluster startup/teardown in sbatch script |

### Stage 3: Config & Cleanup
| File | Change |
|---|---|
| `configs/base.yaml` | Already has `slurm:` and `ray:` sections (no changes needed) |
| `distributed.py` | **Delete** — torch.distributed approach replaced by Ray |

### Stage 4 (Future): Training Integration
| File | Purpose |
|---|---|
| `train_integration/rollout.py` | `@rllm.rollout` wrapper for Path A (UnifiedTrainer) |
| `train_integration/evaluator.py` | `@rllm.evaluator` wrapper (shared by both paths) |
| `train_integration/train.py` | Training script for Path A (UnifiedTrainer) |
| `train_integration/rollout_async.py` | **Path B:** Async rollout function for FullyAsyncTaskRunner |
| `train_integration/train_async.py` | **Path B:** Training script using FullyAsyncTaskRunner |
| `mle_agent/agent_async.py` | **Path B:** Native async `_run_agent_loop_async()` using RolloutClient |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Ray not installed in conda env | `eval_ray.py` imports Ray at top — clear error message. Ray is already a rllm dependency (used by Verl backend). |
| OmegaConf serialization edge cases | `OmegaConf.to_container(cfg, resolve=True)` handles interpolations. Test with all config files. |
| Ray task OOM (sandbox process heavy) | `num_cpus=1` limits concurrency per node. Can add `memory=` resource spec if needed. |
| SLURM Ray cluster startup race | Sleep 10-15s between head start and worker join. Add retry logic if workers fail to connect. |
| Sandbox `manager_uri` points to single node | All nodes must reach the AgentBox manager. Verify network connectivity in multi-node SLURM. This is an existing constraint — sandbox manager runs as a separate service. |
| Large number of concurrent sandboxes | AgentBox manager has container limits. Add `max_concurrent` config to throttle if needed (Ray supports resource-based throttling via custom resources). |

### Fully Async Training Specific Risks (Path B)

| Risk | Mitigation |
|---|---|
| `RolloutClient` vs OpenAI interface mismatch | Stage 4d adds `_run_agent_loop_async()` that uses RolloutClient directly. Eval still uses sync OpenAI client. |
| MLE-bench rollouts are slow (sandbox heavy) | Fully async helps — training continues while rollouts generate. Staleness threshold allows 10%+ extra samples. |
| Weight sync interrupts long rollouts | `RolloutClient.resume_event` handles pause/resume. Aborted requests auto-retry with new weights. |
| Tokenizer dependency for RolloutClient | Pass tokenizer to `mle_bench_rollout_fn()`. Use same tokenizer as SGLang inference servers. |
| Sandbox state during weight sync | Sandbox is independent of LLM weights — no reset needed. Only in-flight LLM calls are aborted. |
| Long evaluation time (code mode) | Evaluation happens after rollout completes. Consider running eval async or in separate actor. |
