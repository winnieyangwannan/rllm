# MLE-bench Fully Async Training Plan

Plan for integrating MLE-bench with rLLM's fully async training infrastructure.

**Reference implementation:** `examples/fully_async/deepresearch/`

---

## Architecture Overview

Fully async mode **decouples inference (rollout) from training** — they run concurrently as separate Ray actors connected via an async MessageQueue.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FullyAsyncTaskRunner (orchestrator)                   │
│                              @ray.remote(num_cpus=1)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌─────────────────┐  trajectories  ┌──────────────┐  gradients   ┌──────────────────┐
│ RolloutExecutor │───────────────▶│ MessageQueue │─────────────▶│ FullyAsyncTrainer│
│ @ray.remote     │                │ @ray.remote  │              │ @ray.remote      │
│ (num_cpus=10)   │                │ (num_cpus=2) │              │ (num_cpus=10)    │
└────────┬────────┘                └──────────────┘              └────────┬─────────┘
         │ HTTP                                                           │
         ▼                                                                │
┌─────────────────┐     pause/resume/sync_weights     ┌───────────────────┴────────┐
│InferenceManager │◀──────────────────────────────────│  ParameterSynchronizer     │
│ (SGLang servers)│              NCCL                 │  @ray.remote               │
│ @ray.remote     │◀══════════════════════════════════│  (broadcast actor weights) │
│ (num_cpus=10)   │                                   └────────────────────────────┘
└─────────────────┘
```

---

## What Needs to Be Built

The fully async infrastructure (`rllm/experimental/fully_async/`) is **already built** and is generic.
We only need to provide:

1. **An async agent loop** that uses `RolloutClient` instead of `openai.OpenAI`
2. **A rollout function** matching the required signature
3. **A training script** using `AsyncAgentTrainer`
4. **A registered dataset** for `DatasetRegistry`

Everything else (RolloutExecutor, MessageQueue, FullyAsyncTrainer, ParameterSynchronizer,
InferenceManager, weight sync, staleness management) is handled by the framework.

---

## Key Differences from Deep Research Example

| Aspect | Deep Research | MLE-bench |
|--------|--------------|-----------|
| Tool execution | HTTP call to RAG server (async, fast) | `sandbox.exec()` in container (sync, slow) |
| Rollout duration | Seconds to minutes | Minutes to hours |
| Side effects | None (stateless search) | Stateful sandbox (files, processes) |
| Reward | F1 score (fast, in-process) | Percentile via MLEEvaluator (slow, needs sandbox) |
| Tools | 1 tool (`local_search`) | 5 tools (`bash`, `edit`, `create`, `submit`, `check_submission_validity`) |
| Concurrency constraint | Limited by inference throughput | Limited by sandbox container pool |

---

## Implementation Plan

### File Structure

```
examples/mlebench/
├── train.py              # Entry point (like deepresearch/train.py)
├── agent.py              # Async agent loop (like deepresearch/search_agent.py)
├── configs/
│   └── fully_async.yaml  # Hydra config overrides
└── scripts/
    └── register_dataset.py  # One-time dataset registration
```

### Step 1: Register MLE-bench Dataset

The `RolloutExecutor` loads data via `DatasetRegistry.load_dataset(name, split)`.
Each item becomes `**kwargs` to `rollout_fn`. Must register a dataset where each row
contains the fields the rollout function needs.

```python
# scripts/register_dataset.py
import json
from rllm.data.dataset import DatasetRegistry

def register_mlebench_dataset(task_jsonl_path: str, split: str = "train"):
    """Register MLE-bench tasks as an rLLM dataset.

    Each row should have at minimum:
    - instance_id: str (e.g. "mlsp-2013-birds")
    - task_description: str
    """
    data = []
    with open(task_jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            data.append(row)

    DatasetRegistry.register_dataset(
        name="mlebench",
        data=data,
        split=split,
        source="mlebench",
        description="MLE-bench Kaggle competition tasks",
        category="code",
    )
    print(f"Registered {len(data)} tasks as 'mlebench' split='{split}'")
```

**Config reference:**
```yaml
data:
  train_dataset_name: mlebench
  train_split: train
  # val split can be a held-out subset of tasks
  val_dataset_name: mlebench
  val_split: val
```

---

### Step 2: Async Agent Loop

**File:** `examples/mlebench/agent.py`

This is the core rewrite. Must replace `openai.OpenAI` with `RolloutClient` and
build token-level `Sequence`/`Trajectory` objects for training.

**Critical types:**
- `rllm.experimental.fully_async.protocol.Trajectory` — list of `Sequence`s + reward
- `rllm.experimental.fully_async.protocol.Sequence` — prompt_ids + response_ids + logprobs + masks
- `rllm.experimental.fully_async.client.RolloutClient` — async HTTP client with pause/resume

These are **NOT** the same as `rllm.types.Trajectory` (which is for eval, not training).

```python
# examples/mlebench/agent.py

import asyncio
import json
import time

from rllm.experimental.fully_async.client import RolloutClient
from rllm.experimental.fully_async.protocol import Trajectory

# Import MLE-bench tools (reuse existing schemas)
from mle_agent.tools import get_tools, execute_tool

# Sampling defaults
TEMP = 1.0
TOPP = 1.0
OVERLONG_FILTER = True


class MLEBenchAgent:
    """Async MLE-bench agent using RolloutClient for fully async training.

    Modeled on deepresearch/search_agent.py::SearchAgent.
    """

    def __init__(
        self,
        client: RolloutClient,
        sandbox,
        max_turns: int = 128,
        session_timeout: float = 360.0,
        rollout_timeout: float = 32400.0,
        temperature: float = TEMP,
        top_p: float = TOPP,
        submit_file: str = "csv",
        check_submission_validity: bool = True,
        task_id: str = "",
        mle_bench_data_dir: str = "",
    ):
        self.llm = client
        self.sandbox = sandbox
        self.max_turns = max_turns
        self.session_timeout = session_timeout
        self.rollout_timeout = rollout_timeout
        self.temperature = temperature
        self.top_p = top_p
        self.submit_file = submit_file
        self.task_id = task_id
        self.mle_bench_data_dir = mle_bench_data_dir

        # Get tool schemas (OpenAI function-calling format)
        # These are passed to tokenizer.apply_chat_template(tools=...) inside RolloutClient
        self.tool_schemas = get_tools(
            submit_file=submit_file,
            check_submission_validity=check_submission_validity,
        )

    async def generate(self, messages):
        """Single LLM call via RolloutClient.

        Returns:
            (message_dict, num_tokens, output) — same pattern as SearchAgent.generate()
        """
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1,
            "repetition_penalty": 1.0,
        }

        # RolloutClient.chat_completion handles:
        # - tokenizer.apply_chat_template(messages, tools=self.tool_schemas)
        # - HTTP to SGLang server
        # - pause/resume during weight sync
        # - abort/retry on weight change
        # Returns (message_dict, OutputWithVersion)
        response, output = await self.llm.chat_completion(
            messages,
            sampling_params=sampling_params,
            tools=self.tool_schemas,
        )
        return response, len(output.all_response_ids()), output

    async def exec_tool_call(self, tool_calls: list[dict]) -> tuple[list[dict], dict, str | None]:
        """Execute tool calls in the sandbox.

        Key difference from deepresearch: sandbox.exec() is synchronous and blocking,
        so we wrap it with asyncio.to_thread() to avoid blocking the event loop.

        Returns:
            (tool_messages, metrics, solution_content)
            - tool_messages: list of tool response dicts
            - metrics: dict of numeric metrics (float-safe)
            - solution_content: train.py content if submit was called, else None
        """
        metrics = {"tool_calls": 0, "tool_errors": 0, "tool_time": 0.0}
        solution_content = None  # Track separately (not in metrics dict)

        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
                metrics["parse_error"] = 0
            except Exception:
                args = {}
                metrics["parse_error"] = metrics.get("parse_error", 0) + 1

            # Execute tool in a thread to avoid blocking async event loop.
            # sandbox.exec() is synchronous — runs bash in a container.
            # Wrap in wait_for to prevent indefinite hangs if the thread never returns.
            tool_start = time.time()
            try:
                tool_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        execute_tool,
                        sandbox=self.sandbox,
                        tool_name=tool_name,
                        args=args,
                        task_id=self.task_id,
                        session_timeout=self.session_timeout,
                        mle_bench_data_dir=self.mle_bench_data_dir,
                        submit_file=self.submit_file,
                    ),
                    timeout=self.session_timeout + 60,  # Grace period beyond tool timeout
                )
            except asyncio.TimeoutError:
                tool_result = (f"Tool '{tool_name}' timed out after {self.session_timeout + 60}s", False, None)
                metrics["tool_errors"] += 1
            tool_time = time.time() - tool_start

            metrics["tool_calls"] += 1
            metrics["tool_time"] += tool_time

            # execute_tool returns (output_str, is_terminal, solution_content)
            output_str, is_terminal, sol_content = tool_result

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": output_str,
            })

            # If submit tool was called, capture solution and signal done
            if is_terminal:
                solution_content = sol_content
                metrics["is_terminal"] = 1.0  # Use float for sanitization
                break

        return tool_messages, metrics, solution_content

    async def run(self, messages: list[dict]) -> dict:
        """Run the multi-turn agent loop.

        Returns dict with:
        - trajectory: protocol.Trajectory (token-level, for training)
        - messages: list[dict] (conversation history, for logging)
        - pred_solution: str | None
        - metrics: dict
        """
        trajectory = Trajectory(sequences=[], reward=0, metadata={})
        pred_solution = None
        overlong = False
        termination_reason = "max_turns"
        start_time = time.time()
        total_completion_tokens = 0
        total_tool_calls = 0
        total_tool_time = 0.0
        turn = 0

        try:
            for turn in range(self.max_turns):
                # Check rollout timeout
                elapsed = time.time() - start_time
                if elapsed >= self.rollout_timeout:
                    termination_reason = "rollout_timeout"
                    break

                # --- LLM call ---
                try:
                    response, completion_tokens, output = await self.generate(messages)
                except Exception as e:
                    print(f"LLM call failed on turn {turn}: {e}")
                    termination_reason = "model_call_error"
                    break

                messages.append(response)

                # Append Sequence to Trajectory (token-level data for training)
                # output.to_sequence() creates Sequence with prompt_ids, response_ids,
                # response_logprobs, response_masks, start_version, end_version
                trajectory.append(output.to_sequence())
                total_completion_tokens += completion_tokens

                # --- Tool execution ---
                tool_calls = response.get("tool_calls", [])
                if not tool_calls:
                    # No tool calls = format error (MLE agent always uses tools)
                    # Could retry, but for training let it count as a bad trajectory
                    termination_reason = "no_tool_calls"
                    break

                if turn < self.max_turns - 1:
                    tool_messages, tool_metrics, sol_content = await self.exec_tool_call(tool_calls)
                    messages.extend(tool_messages)
                    total_tool_calls += tool_metrics.get("tool_calls", 0)
                    total_tool_time += tool_metrics.get("tool_time", 0.0)

                    # Check if submit was called
                    if tool_metrics.get("is_terminal"):
                        pred_solution = sol_content  # Captured from exec_tool_call return
                        termination_reason = "submit"
                        break
                else:
                    # Last turn, no more tool calls allowed
                    break

        except Exception as e:
            import traceback
            overlong = True
            print(f"Error during agent run: {e}")
            traceback.print_exc()
            termination_reason = "error"

        # --- Mask overlong/errored trajectories ---
        # These don't participate in loss calculation (but still in advantage)
        if OVERLONG_FILTER and overlong:
            for seq in trajectory.sequences:
                seq.response_masks = [0] * len(seq.response_masks)

        aggregated_metrics = {
            "num_turns": min(turn + 1, self.max_turns),
            "total_tool_calls": total_tool_calls,
            "total_tool_time": total_tool_time,
            "total_completion_tokens": total_completion_tokens,
            "rollout_duration": time.time() - start_time,
            "overlong": overlong,
            "termination_reason": termination_reason,
            "merged_steps": len(trajectory.merge()),
        }

        return {
            "trajectory": trajectory,
            "messages": messages,
            "pred_solution": pred_solution,
            "metrics": aggregated_metrics,
        }
```

**Key design decisions:**

1. **`asyncio.to_thread(execute_tool, ...)`** — The sandbox calls (`sandbox.exec()`) are
   synchronous and can take minutes (bash commands with timeout). Wrapping in `to_thread`
   prevents blocking the async event loop so other rollouts can proceed concurrently.

2. **`output.to_sequence()`** — Each LLM call produces a `Sequence` with `prompt_ids`,
   `response_ids`, `response_logprobs`, and `response_masks`. These accumulate in the
   `Trajectory`. The trainer merges overlapping sequences (multi-turn conversation where
   each turn's prompt includes all previous turns) via `Trajectory.merge()`.

3. **Tool schemas go to `client.chat_completion(tools=...)`** — The `RolloutClient` passes
   these to `tokenizer.apply_chat_template(messages, tools=tools)` for the model being
   trained. This means the model must have a chat template that supports tool calling
   (e.g., Qwen3 does).

4. **Error/overlong masking** — When a rollout fails, we zero out `response_masks` so those
   tokens don't contribute to the policy gradient loss. The trajectory is still included
   (for advantage normalization) but produces zero gradient.

---

### Step 3: Rollout Function and Reward

**File:** `examples/mlebench/train.py`

The rollout function is the glue between the framework and the agent. It must:
1. Create a sandbox for this task
2. Run the agent loop
3. Compute reward (percentile via MLEEvaluator)
4. Return a `protocol.Trajectory` with reward set

```python
# examples/mlebench/train.py

import asyncio
import random
import time

import hydra

from rllm.experimental.fully_async.runner import AsyncAgentTrainer

from .agent import MLEBenchAgent, OVERLONG_FILTER


# ============================================================================
# Module-level config (captured from Hydra, used by rollout_fn closures)
# Similar pattern to deepresearch/train.py using module-level tool objects
# ============================================================================

# These will be set by main() before training starts
AGENT_CONFIG = {}  # Populated from Hydra config

# Sandbox concurrency semaphore — limits concurrent containers
# Set in main() based on config
SANDBOX_SEMAPHORE: asyncio.Semaphore | None = None


async def _create_sandbox_with_retry(task_data: dict, max_retries: int = 3, retry_delay: float = 5.0, creation_timeout: float = 120.0):
    """Create sandbox with retry logic for transient failures.

    Container creation can fail due to resource exhaustion or manager overload.
    Retrying with backoff helps handle transient issues.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_create_sandbox, task_data),
                timeout=creation_timeout,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"[Sandbox] Creation failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"[Sandbox] Creation failed after {max_retries} attempts: {e}")
    raise last_error


def _create_sandbox(task_data: dict):
    """Create an AgentBox sandbox for a single rollout.

    Returns sandbox instance. Caller must close it.
    """
    from agentbox import ContainerConfig
    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    cfg = AGENT_CONFIG
    task_id = task_data["instance_id"]
    data_path = f"{cfg['mle_bench_data_dir']}/{task_id}/prepared/public"

    container_config = ContainerConfig(
        superimage_directory=cfg["superimage_directory"],
        superimage_version=cfg["superimage_version"],
        container_runtime="apptainer",
        read_only_overlays=[cfg["superimage_overlay"]],
        read_only_binds={data_path: "/root/data"},
        working_dir="/workspace",
        env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
    )

    sandbox = AgentBoxSandbox(
        name=f"train-{task_id}-{random.randint(0, 99999)}",
        manager_uri=cfg["manager_uri"],
        container_config=container_config,
    )
    # Initialize workspace (sync call, OK in to_thread context)
    sandbox.exec("mkdir -p /workspace")

    return sandbox


def _compute_reward(pred_solution: str | None, sandbox, task_id: str) -> tuple[float, dict]:
    """Compute reward using MLEEvaluator.

    This is synchronous and potentially slow (especially in code mode where
    it executes train.py). Will be called inside asyncio.to_thread().

    Returns:
        (reward_float, reward_metadata_dict)
    """
    from mle_agent.evaluator import MLEEvaluator

    cfg = AGENT_CONFIG

    evaluator = MLEEvaluator(
        mle_bench_data_dir=cfg["mle_bench_data_dir"],
        eval_timeout=cfg.get("eval_timeout", 32400),
        submit_file=cfg.get("submit_file", "csv"),
    )

    class MockEpisode:
        def __init__(self, pred_solution, sandbox):
            self.artifacts = {
                "_sandbox": sandbox,
                "pred_solution": pred_solution,
            }

    task = {"task_id": task_id, "instance_id": task_id}
    episode = MockEpisode(pred_solution, sandbox)

    try:
        eval_output = evaluator.evaluate(task, episode)
        signals_dict = {s.name: s.value for s in eval_output.signals}

        return eval_output.reward, {
            "percentile": eval_output.reward,
            "raw_score": signals_dict.get("raw_score"),
            "valid_submission": signals_dict.get("valid_submission", 0.0),
            "is_correct": eval_output.reward > 0,
        }
    except Exception as e:
        print(f"Evaluation failed for {task_id}: {e}")
        return 0.0, {"eval_error": str(e), "is_correct": False}


def _sanitize_metrics(metrics: dict) -> dict:
    """Keep only float-convertible values and selected string fields for logging."""
    ALLOWED_STRING_KEYS = {"termination_reason"}
    sanitized = {}
    for k, v in metrics.items():
        if k in ALLOWED_STRING_KEYS and isinstance(v, str):
            sanitized[k] = v
            continue
        try:
            sanitized[k] = float(v)
        except (ValueError, TypeError):
            continue
    return sanitized


# ============================================================================
# Rollout functions (match deepresearch/train.py pattern)
# ============================================================================

async def rollout_fn(client, tokenizer, **kwargs):
    """Training rollout function.

    Signature: async def(client: RolloutClient, tokenizer, **kwargs) -> Trajectory
    **kwargs comes from the dataset row (instance_id, task_description, etc.)
    """
    start_time = time.time()
    param_version_start = client.cur_version

    cfg = AGENT_CONFIG
    task_data = kwargs
    task_id = task_data["instance_id"]

    # Timing breakdown for metrics
    timing = {}
    sandbox_end_time = start_time
    agent_end_time = start_time
    eval_start_time = start_time

    # --- Acquire sandbox slot (limits concurrent containers) ---
    if SANDBOX_SEMAPHORE is not None:
        await SANDBOX_SEMAPHORE.acquire()

    sandbox = None
    try:
        # --- Create sandbox with retry (sync, run in thread) ---
        sandbox_start = time.time()
        sandbox = await _create_sandbox_with_retry(task_data)
        timing["timing_s/sandbox_create"] = time.time() - sandbox_start

        # --- Build initial messages ---
        if cfg.get("submit_file") == "code":
            from mle_agent.prompts_code import INSTANCE_PROMPT, SYSTEM_PROMPT
            system_prompt = SYSTEM_PROMPT.format(
                timeout_min=int(cfg.get("session_timeout", 360) / 60),
                context_size=cfg.get("context_size", 131072),
                eval_timeout_hrs=int(cfg.get("rollout_timeout", 32400) / 3600),
                max_turns=cfg.get("max_turns", 128),
            )
        else:
            from mle_agent.prompts_csv import INSTANCE_PROMPT, SYSTEM_PROMPT
            system_prompt = SYSTEM_PROMPT.format(
                timeout_min=int(cfg.get("session_timeout", 360) / 60),
                timeout_unit="minutes",
                context_size=cfg.get("context_size", 131072),
                rollout_timeout_hrs=int(cfg.get("rollout_timeout", 32400) / 3600),
                max_turns=cfg.get("max_turns", 128),
            )

        instance_prompt = INSTANCE_PROMPT.format(
            task_description=task_data.get("task_description", ""),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        # --- Run agent ---
        agent = MLEBenchAgent(
            client=client,
            sandbox=sandbox,
            max_turns=cfg.get("max_turns", 128),
            session_timeout=cfg.get("session_timeout", 360),
            rollout_timeout=cfg.get("rollout_timeout", 32400),
            temperature=cfg.get("temperature", 1.0),
            submit_file=cfg.get("submit_file", "csv"),
            check_submission_validity=cfg.get("check_submission_validity", True),
            task_id=task_id,
            mle_bench_data_dir=cfg.get("mle_bench_data_dir", ""),
        )

        sandbox_end_time = time.time()
        result = await agent.run(messages)
        agent_end_time = time.time()
        trajectory = result["trajectory"]
        pred_solution = result["pred_solution"]
        agent_metrics = result["metrics"]

        # --- Compute reward (sync, run in thread) ---
        # MLEEvaluator is synchronous and can be slow (especially in code mode)
        eval_start_time = time.time()
        try:
            reward, reward_meta = await asyncio.wait_for(
                asyncio.to_thread(_compute_reward, pred_solution, sandbox, task_id),
                timeout=cfg.get("eval_timeout", 32400) + 120,  # Grace period beyond eval timeout
            )
        except asyncio.TimeoutError:
            print(f"[Eval] Timed out for {task_id}")
            reward, reward_meta = 0.0, {"eval_error": "timeout", "is_correct": False}

        # --- Mask trajectories for error cases ---
        # No submission = reward 0, but still train on the trajectory
        # (model should learn that not submitting is bad)
        # Only mask on actual errors (sandbox crash, overlong, etc.)
        if agent_metrics.get("termination_reason") in ("error", "model_call_error"):
            for seq in trajectory.sequences:
                seq.response_masks = [0] * len(seq.response_masks)

        trajectory.reward = reward
        timing["timing_s/agent_run"] = agent_end_time - sandbox_end_time
        timing["timing_s/eval"] = time.time() - eval_start_time

    finally:
        # Always close sandbox to free container resources
        if sandbox is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(sandbox.close),
                    timeout=60.0,  # Don't wait forever for cleanup
                )
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[Sandbox] Close failed for {task_id}: {e}")

        # Release sandbox semaphore slot
        if SANDBOX_SEMAPHORE is not None:
            SANDBOX_SEMAPHORE.release()

    # --- Build metadata ---
    end_time = time.time()
    param_version_end = client.cur_version

    metadata = {
        "processing_time": end_time - start_time,
        "param_version_start": param_version_start,
        "param_version_end": param_version_end,
        "param_version": param_version_end,
        "is_partial": param_version_start != param_version_end,
    }

    # Merge agent metrics, reward metadata, and timing
    agent_metrics.update(reward_meta)
    agent_metrics.update(metadata)
    agent_metrics.update(timing)
    agent_metrics = _sanitize_metrics(agent_metrics)

    trajectory.metadata = agent_metrics

    if random.random() < 0.01:
        print(f"\n{'=' * 70}")
        print(f"Task: {task_id}")
        print(f"Reward: {reward:.4f}")
        print(f"Termination: {result['metrics'].get('termination_reason')}")
        print(f"Turns: {result['metrics'].get('num_turns')}")
        print(f"Duration: {end_time - start_time:.1f}s")
        print(f"Param versions: {param_version_start} -> {param_version_end}")
        print(f"{'=' * 70}\n")

    return trajectory


async def val_rollout_fn(client, tokenizer, **kwargs):
    """Validation rollout function.

    Same as rollout_fn but shares the SANDBOX_SEMAPHORE, meaning validation
    competes with training rollouts for container slots. If validation should
    not starve training, consider a dedicated VAL_SANDBOX_SEMAPHORE with a
    small limit (e.g., 2-4 slots) or running validation only when training
    rollouts are paused for weight sync.
    """
    return await rollout_fn(client, tokenizer, **kwargs)


# ============================================================================
# Entry point
# ============================================================================

@hydra.main(
    config_path="pkg://rllm.experimental.fully_async.config",
    config_name="fully_async_ppo_trainer",
    version_base=None,
)
def main(config):
    """Main entry point for MLE-bench fully async training.

    Usage:
        python -m examples.mlebench.train \
            data.train_dataset_name=mlebench \
            data.train_split=train \
            +mlebench.submit_file=csv \
            +mlebench.mle_bench_data_dir=/path/to/data \
            +mlebench.manager_uri=http://localhost:8080 \
            +mlebench.superimage_directory=/path/to/superimage \
            +mlebench.superimage_version=v1 \
            +mlebench.superimage_overlay=/path/to/overlay
    """
    global AGENT_CONFIG, SANDBOX_SEMAPHORE

    # Extract MLE-bench specific config into module-level dict
    # This is the same pattern as deepresearch using module-level tool objects
    mlebench_cfg = config.get("mlebench", {})
    AGENT_CONFIG = {
        "submit_file": mlebench_cfg.get("submit_file", "csv"),
        "mle_bench_data_dir": mlebench_cfg.get("mle_bench_data_dir", ""),
        "manager_uri": mlebench_cfg.get("manager_uri", ""),
        "superimage_directory": mlebench_cfg.get("superimage_directory", ""),
        "superimage_version": mlebench_cfg.get("superimage_version", ""),
        "superimage_overlay": mlebench_cfg.get("superimage_overlay", ""),
        "max_turns": mlebench_cfg.get("max_turns", 128),
        "session_timeout": mlebench_cfg.get("session_timeout", 360),
        "rollout_timeout": mlebench_cfg.get("rollout_timeout", 32400),
        "eval_timeout": mlebench_cfg.get("eval_timeout", 32400),
        "temperature": mlebench_cfg.get("temperature", 1.0),
        "context_size": mlebench_cfg.get("context_size", 131072),
        "check_submission_validity": mlebench_cfg.get("check_submission_validity", True),
    }

    # Initialize sandbox concurrency limiter
    # This caps concurrent containers independent of LLM rollout concurrency
    max_sandboxes = mlebench_cfg.get("max_concurrent_sandboxes", 32)
    SANDBOX_SEMAPHORE = asyncio.Semaphore(max_sandboxes)
    print(f"[MLE-bench] Sandbox concurrency limit: {max_sandboxes}")

    # Health check: verify sandbox manager is reachable before starting training
    _verify_sandbox_manager(AGENT_CONFIG["manager_uri"])

    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,
        val_rollout_fn=val_rollout_fn,
    )
    trainer.train()


def _verify_sandbox_manager(manager_uri: str):
    """Verify sandbox manager is reachable before starting training.

    Fails fast with a clear error message if manager is down.
    """
    import httpx

    if not manager_uri:
        print("[WARNING] No manager_uri configured, skipping health check")
        return

    try:
        # Most AgentBox managers have a /health or / endpoint
        response = httpx.get(f"{manager_uri}/health", timeout=10.0)
        if response.status_code == 200:
            print(f"[MLE-bench] Sandbox manager healthy: {manager_uri}")
        else:
            print(f"[WARNING] Sandbox manager returned {response.status_code}: {manager_uri}")
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Cannot reach sandbox manager at {manager_uri}: {e}\n"
            "Make sure the AgentBox manager is running before starting training."
        )


if __name__ == "__main__":
    main()
```

---

### Step 4: Handling MLE-bench-Specific Challenges

#### 4a. Sandbox Lifecycle and Concurrency

Each rollout creates and destroys a container sandbox. With hundreds of concurrent rollouts,
this is the primary resource bottleneck.

**Concurrency is controlled by two mechanisms:**
1. `RolloutExecutor.max_concurrent_rollout` — limits total in-flight rollouts (default: 128 * num_servers)
2. `asyncio.Semaphore` inside `RolloutExecutor.fit()` — enforces the limit

For MLE-bench, this needs to be much lower than deep research because each sandbox
is heavyweight (container with GPU). Override in config:

```yaml
# Much lower concurrency than deep research
# Each MLE-bench rollout needs a container with GPU
rollout:
  # This controls inference server count (SGLang replicas)
  n_gpus_per_node: 4
  nnodes: 1

# The effective concurrency is also limited by sandbox manager capacity
# Tune based on your AgentBox manager's limits
```

**Cleanup on abort:** When `ParameterSynchronizer` syncs weights, in-flight LLM calls
get `finish_reason="abort"`. The `RolloutClient` automatically retries the LLM call.
The sandbox is unaffected — it persists across LLM calls within a rollout. This is safe
because the sandbox state from previous turns is still valid (same as normal continuation).

The `try/finally` in `rollout_fn` ensures `sandbox.close()` is always called, even if
the rollout errors out.

#### 4b. Blocking I/O in Async Context

MLE-bench tools are synchronous (they call `sandbox.exec()` which does HTTP to the
container manager and waits). In the async event loop, these must be wrapped:

```python
# BAD: blocks the event loop, starves other rollouts
tool_result = execute_tool(tool_name, tool_args, sandbox, ...)

# GOOD: runs in a thread, other rollouts can proceed
tool_result = await asyncio.to_thread(execute_tool, tool_name, tool_args, sandbox, ...)
```

Same for sandbox creation, evaluation, and cleanup.

#### 4c. Reward Function

The reward is the **percentile** from `MLEEvaluator.evaluate()`, a float in [0, 1].

- `1.0` = best possible score on the leaderboard
- `0.0` = worst or invalid/no submission

This is computed synchronously and can be slow (code mode executes `train.py` in the container).
It runs via `asyncio.to_thread()`.

**Masking policy:**
| Scenario | Reward | Response masks |
|----------|--------|---------------|
| Normal submission | percentile | Keep (1s) |
| No submission (max_turns/timeout) | 0.0 | Keep — model should learn this is bad |
| Sandbox crash / model error | 0.0 | Zero — don't train on broken trajectories |
| Overlong (context exceeded) | 0.0 | Zero — tokens may be incomplete |

#### 4d. Tool Schema Compatibility

`RolloutClient.chat_completion()` passes `tools` to `tokenizer.apply_chat_template()`.
The existing MLE-bench tool schemas from `get_tools()` are in OpenAI function-calling format:

```python
{"type": "function", "function": {"name": "bash", "parameters": {...}}}
```

This format is compatible with Qwen3's chat template (which supports tool calling).
If using a different model, verify the chat template handles the tool schema format.

#### 4e. Weight Sync During Long Rollouts

MLE-bench rollouts can take hours. During this time, multiple weight syncs may occur.
Each sync pauses the `RolloutClient` briefly, then resumes. Key implications:

- A single trajectory may span multiple param versions (tracked in `Sequence.start_version`
  and `Sequence.end_version`)
- The `is_partial` flag in metadata indicates a version change happened during rollout
- The staleness threshold controls how many "stale" samples the trainer accepts
- For very long rollouts, consider setting `staleness_threshold` higher than default

```yaml
async_training:
  staleness_threshold: 3    # Allow more staleness for long MLE-bench rollouts
  trigger_parameter_sync_step: 4
  required_samples: 32      # Smaller batches (fewer concurrent MLE-bench tasks)
```

---

## Data Flow Summary

```
1. DatasetRegistry yields task dict {instance_id, task_description, ...}
       │
       ▼
2. RolloutExecutor dispatches → rollout_fn(client, tokenizer, **task_dict)
       │
       ▼
3. rollout_fn:
   a. Create AgentBox sandbox (asyncio.to_thread)
   b. Build system/user prompt from task_description
   c. Run MLEBenchAgent.run(messages):
      │
      │  for turn in range(max_turns):
      │    ├── await client.chat_completion(messages, tools=tool_schemas)
      │    │   → RolloutClient → HTTP → SGLang → response + OutputWithVersion
      │    │   → output.to_sequence() → Sequence(prompt_ids, response_ids, logprobs)
      │    │   → trajectory.append(sequence)
      │    │
      │    └── await asyncio.to_thread(execute_tool, ...)
      │        → sandbox.exec(command) → tool output
      │        → append to messages
      │        → if submit → break
      │
   d. Compute reward via MLEEvaluator (asyncio.to_thread)
   e. Close sandbox (asyncio.to_thread)
   f. Return Trajectory with sequences, reward, metadata
       │
       ▼
4. RolloutExecutor groups n trajectories → TrajectoryGroup → MessageQueue
       │
       ▼
5. FullyAsyncTrainer pulls batch, computes advantages (GRPO), updates actor
       │
       ▼
6. ParameterSynchronizer broadcasts new weights to SGLang servers
       │
       ▼
7. Next rollouts use updated model (in-flight rollouts continue with old weights)
```

---

## Config Template

```yaml
# configs/fully_async.yaml
# Override on top of rllm/experimental/fully_async/config/fully_async_ppo_trainer.yaml

# Actor/Rollout model config (verl structure)
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-8B  # Or your model
  rollout:
    n: 1  # Single rollout per task (MLE-bench rollouts are expensive)
    tensor_model_parallel_size: 1

# Data
data:
  train_dataset_name: mlebench
  train_split: train
  val_dataset_name: mlebench
  val_split: val
  max_prompt_length: 4096
  max_response_length: 32768

# Rollout (inference servers)
rollout:
  n_gpus_per_node: 4
  nnodes: 1
  total_rollout_steps: null  # Auto-calculated from dataset size

# Async training
async_training:
  staleness_threshold: 3          # Higher for long MLE-bench rollouts
  trigger_parameter_sync_step: 4
  required_samples: 32            # Smaller than deepresearch (rollouts are slower)
  require_batches: 1
  compute_prox_log_prob: false

# Trainer
trainer:
  total_epochs: 1
  save_freq: 10
  val_before_train: true
  device: cuda  # or npu

# MLE-bench specific
mlebench:
  submit_file: csv
  mle_bench_data_dir: /path/to/mlebench/data
  manager_uri: http://localhost:8080
  superimage_directory: /path/to/superimage
  superimage_version: v1
  superimage_overlay: /path/to/overlay
  max_turns: 128
  session_timeout: 360
  rollout_timeout: 32400
  eval_timeout: 32400
  temperature: 1.0
  context_size: 131072
  check_submission_validity: true
  max_concurrent_sandboxes: 32    # Limits concurrent containers (independent of LLM concurrency)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| **New files** | |
| `examples/mlebench/train.py` | Entry point, rollout_fn, reward computation |
| `examples/mlebench/agent.py` | `MLEBenchAgent` async agent loop |
| `examples/mlebench/configs/fully_async.yaml` | Hydra config overrides |
| `examples/mlebench/scripts/register_dataset.py` | One-time dataset registration |
| **Existing (reuse as-is)** | |
| `agenthub/mle_agent/mle_agent/tools.py` | Tool schemas + `execute_tool()` |
| `agenthub/mle_agent/mle_agent/prompts_csv.py` | System/instance prompts (csv mode) |
| `agenthub/mle_agent/mle_agent/prompts_code.py` | System/instance prompts (code mode) |
| `agenthub/mle_agent/mle_agent/evaluator.py` | `MLEEvaluator` for reward |
| **Framework (don't modify)** | |
| `rllm/experimental/fully_async/runner.py` | `AsyncAgentTrainer` |
| `rllm/experimental/fully_async/client.py` | `RolloutClient` |
| `rllm/experimental/fully_async/protocol.py` | `Trajectory`, `Sequence`, `OutputWithVersion` |
| `rllm/experimental/fully_async/rollout_executor.py` | `RolloutExecutor` |
| `rllm/experimental/fully_async/fully_async_trainer.py` | `FullyAsyncTrainer` |
| `rllm/experimental/fully_async/message_queue.py` | `MessageQueue` |
| `rllm/experimental/fully_async/param_sync.py` | `ParameterSynchronizer` |

---

## Open Questions / TODOs

1. ~~**Sandbox concurrency limits:**~~ ✅ Resolved — Added `SANDBOX_SEMAPHORE` with configurable
   `max_concurrent_sandboxes` to limit containers independently of LLM rollout concurrency.

2. ~~**`execute_tool` return signature:**~~ ✅ Verified — `tools.py:execute_tool()` returns
   `(output_str, is_terminal, solution_content)` as assumed. Updated `exec_tool_call` to
   return `solution_content` separately (not in metrics dict) to survive sanitization.

3. **Validation dataset:** Should use a held-out set of tasks, registered as
   `mlebench` split `val`. Define which tasks are train vs val.

4. **Context size with tool calling:** Qwen3's chat template with tools adds overhead
   to the prompt. The `max_prompt_length + max_response_length` in config must account
   for multi-turn conversations that grow over 100+ turns.

5. ~~**n rollouts per prompt:**~~ ✅ Resolved — Config explicitly sets `n: 1` in
   `actor_rollout_ref.rollout.n`. Cross-task normalization is used instead of per-task
   multiple rollouts.

6. **Code mode evaluation timing:** In code mode, `MLEEvaluator` executes `train.py` inside
   the container after the agent finishes. This can take hours. Consider whether evaluation
   should be async or moved to a separate process to avoid blocking other rollouts.

7. **Checkpoint/resume for long rollouts:** Multi-hour rollouts may be interrupted by crashes.
   Consider periodic workspace snapshots or implementing rollout-level checkpointing.

8. **AgentBox lifecycle:** Verify that `AgentBoxSandbox` supports the expected lifecycle:
   - Constructor creates the container
   - `exec()` runs commands
   - `close()` destroys the container
   Add `start()` call if needed.

9. **`MockEpisode` fragility in `_compute_reward`:** The `MockEpisode` class couples to
   the internal API of `MLEEvaluator.evaluate()` — it assumes the evaluator accesses
   `episode.artifacts["_sandbox"]` and `episode.artifacts["pred_solution"]`. If the
   evaluator interface changes, this breaks silently. Consider adding an assertion or
   integration test that verifies the evaluator contract, or refactoring to use the
   evaluator's public API directly.

10. **Validation competing for sandbox pool:** `val_rollout_fn` shares `SANDBOX_SEMAPHORE`
    with training rollouts. During validation, training rollouts may starve or vice versa.
    Consider a dedicated validation semaphore with a small limit (2-4 slots), or scheduling
    validation only during weight sync pauses when training rollouts are already paused.

---

## Complexity Analysis

### Most Complicated Parts (Ranked by Difficulty)

| Rank | Component | Why It's Hard | Risk |
|------|-----------|---------------|------|
| 1 | **Sandbox concurrency & lifecycle** | Containers are stateful, slow to create/destroy, can fail silently | Deadlocks, resource leaks, crashes |
| 2 | **Long rollout resilience** | Multi-hour rollouts amplify every edge case: `sandbox.exec()` hangs, network failures to manager mid-rollout, OOM in container during eval. Every `to_thread` call needs a `wait_for` timeout or the async task hangs forever | Silent hangs, throughput collapse, leaked resources |
| 3 | **Code mode evaluation timing** | Evaluator runs `train.py` post-agent, can take hours. Blocks other rollouts if not properly async. Needs its own timeout and error handling | Blocks training pipeline, starves sandbox pool |
| 4 | **Agent → RolloutClient migration (tool parsing)** | `chat_completion()` relies on `message_utils.py`'s `parse_response()` and `ToolParser` to convert tokens into tool calls. If the model's tool-calling format doesn't match the parser, tool calls silently fail (no `tool_calls` in response → `no_tool_calls` termination). Tool schemas must also be compatible with `tokenizer.apply_chat_template(tools=...)`. Format mismatches between `chat_completion` output and `execute_tool` input (e.g., `arguments` as string vs dict) cause runtime errors | Silent failures, all rollouts terminate immediately without tool use |
| 5 | **Blocking I/O in async context** | `sandbox.exec()` is sync, must not starve event loop. Solved pattern (`asyncio.to_thread`), but must be applied consistently everywhere including sandbox creation, tool execution, evaluation, and cleanup | Silent hangs, throughput collapse |
| 6 | **Token-level trajectory capture** | `output.to_sequence()` is a one-liner provided by the framework — the `Sequence` with `prompt_ids`, `response_ids`, `logprobs`, and `masks` is constructed automatically. Hard to get wrong if you follow the reference implementation | Training crashes if format wrong, but low likelihood |

### Key Insight

The **infrastructure complexity** (sandboxes, timeouts, async I/O) is harder than the **ML complexity** (trajectory format).
Deep Research's RAG tool is stateless HTTP — MLE-bench's sandbox is stateful containers.
The **tool parsing compatibility** between the trained model and `message_utils.py` is an often-overlooked integration risk that can cause silent failures.

---

## Staged Implementation Plan

### Phase 1: Agent with Mock Sandbox (Day 1)

**Goal:** Prove `MLEBenchAgent` works with `RolloutClient` — no containers needed.
Test multi-turn conversation, multiple tool types, and error paths.

```python
# examples/mlebench/tests/test_mock.py
"""Test MLEBenchAgent with mocked sandbox - no containers needed."""

import asyncio
from transformers import AutoTokenizer
from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion, Trajectory

class MockSandbox:
    """Fake sandbox that returns canned responses."""
    def exec(self, cmd, timeout=60):
        return f"[MOCK] Executed: {cmd[:50]}..."
    def close(self):
        pass
    def fetch_file(self, src, dst):
        return False


class MockRolloutClient:
    """Fake client that simulates a multi-turn conversation.

    Turn 0: bash tool call (explore data)
    Turn 1: bash tool call (run training)
    Turn 2: submit tool call (terminate)

    This exercises the full agent loop, not just the happy-path single-turn.
    """
    cur_version = 0
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    _call_count = 0

    async def chat_completion(self, messages, sampling_params, tools):
        turn = self._call_count
        self._call_count += 1

        if turn == 0:
            response = {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call_{turn}",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "ls /root/data"}'
                    }
                }]
            }
        elif turn == 1:
            response = {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call_{turn}",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "python train.py"}'
                    }
                }]
            }
        else:
            response = {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call_{turn}",
                    "function": {
                        "name": "submit",
                        "arguments": '{"train_path": "/workspace/train.py", "submission_path": "/workspace/submission.csv"}'
                    }
                }]
            }

        output = OutputWithVersion(
            prompt_ids=list(range(100 * (turn + 1))),
            output_chunks=[OutputChunk(
                response_ids=list(range(100, 150)),
                response_logprobs=[-0.1] * 50,
                version=0
            )]
        )
        return response, output


class MockRolloutClientNoTools:
    """Fake client that returns a response with no tool_calls.

    Tests the error path where the model fails to call any tool.
    """
    cur_version = 0
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    async def chat_completion(self, messages, sampling_params, tools):
        response = {
            "role": "assistant",
            "content": "I don't know how to use tools.",
        }
        output = OutputWithVersion(
            prompt_ids=list(range(100)),
            output_chunks=[OutputChunk(
                response_ids=list(range(100, 120)),
                response_logprobs=[-0.5] * 20,
                version=0
            )]
        )
        return response, output


async def test_agent_multi_turn():
    """Test multi-turn agent loop (bash -> bash -> submit)."""
    from examples.mlebench.agent import MLEBenchAgent

    agent = MLEBenchAgent(
        client=MockRolloutClient(),
        sandbox=MockSandbox(),
        max_turns=10,
        task_id="test-task",
    )

    result = await agent.run([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this ML task."},
    ])

    # Validate multi-turn: should have 3 sequences (one per LLM call)
    assert len(result["trajectory"].sequences) == 3, (
        f"Expected 3 sequences (3 turns), got {len(result['trajectory'].sequences)}"
    )
    assert result["metrics"]["termination_reason"] == "submit"
    assert result["metrics"]["num_turns"] == 3

    for seq in result["trajectory"].sequences:
        assert len(seq.response_ids) == len(seq.response_logprobs)
        assert len(seq.response_ids) == len(seq.response_masks)

    print("PASSED: test_agent_multi_turn")


async def test_agent_no_tool_calls():
    """Test error path: model returns no tool_calls."""
    from examples.mlebench.agent import MLEBenchAgent

    agent = MLEBenchAgent(
        client=MockRolloutClientNoTools(),
        sandbox=MockSandbox(),
        max_turns=5,
        task_id="test-task",
    )

    result = await agent.run([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this ML task."},
    ])

    assert result["metrics"]["termination_reason"] == "no_tool_calls"
    assert len(result["trajectory"].sequences) == 1
    assert result["pred_solution"] is None

    print("PASSED: test_agent_no_tool_calls")


async def test_agent_max_turns():
    """Test that agent respects max_turns limit."""
    from examples.mlebench.agent import MLEBenchAgent

    # Client that never submits (always calls bash)
    class NeverSubmitClient:
        cur_version = 0
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        async def chat_completion(self, messages, sampling_params, tools):
            response = {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_bash",
                    "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}
                }]
            }
            output = OutputWithVersion(
                prompt_ids=list(range(50)),
                output_chunks=[OutputChunk(
                    response_ids=list(range(50, 70)),
                    response_logprobs=[-0.2] * 20,
                    version=0
                )]
            )
            return response, output

    agent = MLEBenchAgent(
        client=NeverSubmitClient(),
        sandbox=MockSandbox(),
        max_turns=3,
        task_id="test-task",
    )

    result = await agent.run([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this ML task."},
    ])

    assert result["metrics"]["termination_reason"] == "max_turns"
    assert result["metrics"]["num_turns"] == 3
    assert result["pred_solution"] is None

    print("PASSED: test_agent_max_turns")


if __name__ == "__main__":
    asyncio.run(test_agent_multi_turn())
    asyncio.run(test_agent_no_tool_calls())
    asyncio.run(test_agent_max_turns())
```

**What to verify:**
- [ ] `RolloutClient.chat_completion()` interface matches expectations
- [ ] `output.to_sequence()` produces valid `Sequence` objects
- [ ] Multi-turn conversation accumulates sequences correctly (3 sequences for 3 turns)
- [ ] Error path (no tool calls) terminates cleanly with correct reason
- [ ] `max_turns` limit is respected
- [ ] Tool calling round-trips correctly (parse args, execute, append result)

**Run:** `python examples/mlebench/tests/test_mock.py`

---

### Phase 2: Real RolloutClient, Mock Sandbox (Day 2)

**Goal:** Validate against a real SGLang inference server. This is the critical phase for
catching **tool parsing compatibility** issues — the model must produce tool calls that
`message_utils.py`'s `parse_response()` can parse, and the resulting `tool_calls` format
must match what `execute_tool()` expects.

```python
# examples/mlebench/tests/test_real_client.py
"""Test with real RolloutClient but mock sandbox."""

import asyncio
from transformers import AutoTokenizer
from rllm.experimental.fully_async.client import RolloutClient

async def test_real_client():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    client = RolloutClient(
        router_url="http://localhost:30001",  # Running SGLang server
        tokenizer=tokenizer,
    )
    
    from examples.mlebench.agent import MLEBenchAgent
    from examples.mlebench.tests.test_mock import MockSandbox
    
    agent = MLEBenchAgent(
        client=client,
        sandbox=MockSandbox(),
        max_turns=3,
        task_id="test-task",
    )
    
    result = await agent.run([
        {"role": "system", "content": "You are a helpful ML assistant."},
        {"role": "user", "content": "Run `ls /root/data` to see the data."},
    ])
    
    print(f"Turns: {result['metrics']['num_turns']}")
    print(f"Termination: {result['metrics']['termination_reason']}")
    print(f"Sequences: {len(result['trajectory'].sequences)}")
    
    # Check logprobs are real (not mock)
    for i, seq in enumerate(result["trajectory"].sequences):
        print(f"  Seq {i}: prompt={len(seq.prompt_ids)} response={len(seq.response_ids)}")
        assert all(lp < 0 for lp in seq.response_logprobs), "Logprobs should be negative"
    
    print("✅ Phase 2 PASSED: Real client works")

if __name__ == "__main__":
    asyncio.run(test_real_client())
```

**Prerequisites:** SGLang server running with your model
```bash
python -m sglang.launch_server --model Qwen/Qwen3-8B --port 30001
```

**What to verify:**
- [ ] Client connects to server
- [ ] Tool schemas are accepted by the model's chat template
- [ ] Model produces tool calls that `parse_response()` can parse (not empty `tool_calls`)
- [ ] `tool_call["function"]["arguments"]` is a JSON string (not a dict) — matches `json.loads()` in `exec_tool_call`
- [ ] Logprobs are real floats (not mocked)
- [ ] No hangs on pause/resume

---

### Phase 3: Sandbox Lifecycle Stress Test (Day 3-4, parallel with Phase 4)

**Goal:** Prove sandboxes can be created/destroyed reliably under concurrent load.

**Note:** This phase and Phase 4 test independent components (sandbox lifecycle vs. reward
computation) and can run in parallel to save time.

```python
# examples/mlebench/tests/test_sandbox_concurrency.py
"""Stress test sandbox creation/destruction."""

import asyncio
import time

async def test_sandbox_pool():
    from examples.mlebench.train import _create_sandbox, AGENT_CONFIG
    
    # Configure (normally done by main())
    AGENT_CONFIG.update({
        "mle_bench_data_dir": "/path/to/data",
        "manager_uri": "http://localhost:8080",
        "superimage_directory": "/path/to/superimage",
        "superimage_version": "v1",
        "superimage_overlay": "/path/to/overlay",
    })
    
    semaphore = asyncio.Semaphore(4)  # Max 4 concurrent
    results = {"success": 0, "failed": 0}
    
    async def run_one(task_id):
        async with semaphore:
            start = time.time()
            try:
                sandbox = await asyncio.wait_for(
                    asyncio.to_thread(
                        _create_sandbox, {"instance_id": "digit-recognizer"}
                    ),
                    timeout=120.0,
                )
                # Simulate variable hold time (real rollouts hold for minutes/hours)
                hold_time = random.uniform(5, 30)
                result = await asyncio.to_thread(sandbox.exec, f"sleep {int(hold_time)} && echo hello")
                await asyncio.wait_for(
                    asyncio.to_thread(sandbox.close),
                    timeout=60.0,
                )
                results["success"] += 1
                print(f"  {task_id} completed in {time.time() - start:.1f}s (held {hold_time:.0f}s)")
            except asyncio.TimeoutError:
                results["failed"] += 1
                print(f"  {task_id} timed out")
            except Exception as e:
                results["failed"] += 1
                print(f"  {task_id} failed: {e}")
    
    # Run 10 sandboxes with max 4 concurrent
    tasks = [run_one(f"task-{i}") for i in range(10)]
    await asyncio.gather(*tasks)
    
    print(f"\nResults: {results['success']} success, {results['failed']} failed")
    assert results["failed"] == 0, "Some sandboxes failed!"
    print("✅ Phase 3 PASSED: Sandbox concurrency works")

if __name__ == "__main__":
    asyncio.run(test_sandbox_pool())
```

**What to verify:**
- [ ] Sandboxes create without hanging
- [ ] `close()` actually frees resources (check manager dashboard)
- [ ] Semaphore prevents resource exhaustion
- [ ] No leaked containers after test

---

### Phase 4: Reward Computation (Day 3-4, parallel with Phase 3)

**Goal:** Verify `MLEEvaluator` works with your sandbox setup.

```python
# examples/mlebench/tests/test_reward.py
"""Test reward computation with real sandbox."""

import asyncio

async def test_reward_computation():
    from examples.mlebench.train import _create_sandbox, _compute_reward, AGENT_CONFIG
    
    # Configure
    AGENT_CONFIG.update({...})  # Same as Phase 3
    
    sandbox = await asyncio.to_thread(
        _create_sandbox, {"instance_id": "digit-recognizer"}
    )
    
    try:
        # Create dummy submission files
        await asyncio.to_thread(
            sandbox.exec,
            "echo 'import pandas as pd' > /workspace/train.py"
        )
        await asyncio.to_thread(
            sandbox.exec,
            "echo 'ImageId,Label' > /workspace/submission.csv && "
            "for i in $(seq 1 28000); do echo \"$i,0\" >> /workspace/submission.csv; done"
        )
        
        # Compute reward
        pred_solution = "import pandas as pd"
        reward, meta = await asyncio.to_thread(
            _compute_reward, pred_solution, sandbox, "digit-recognizer"
        )
        
        print(f"Reward: {reward}")
        print(f"Metadata: {meta}")
        
        assert reward >= 0.0 and reward <= 1.0, "Reward should be percentile"
        print("✅ Phase 4 PASSED: Reward computation works")
        
    finally:
        await asyncio.to_thread(sandbox.close)

if __name__ == "__main__":
    asyncio.run(test_reward_computation())
```

**What to verify:**
- [ ] Evaluator can access sandbox
- [ ] `fetch_file()` works (for CSV mode)
- [ ] Percentile is computed correctly
- [ ] Errors return 0.0 reward (not crash)

---

### Phase 4.5: Real LLM + Real Sandbox, Limited Turns (Day 5)

**Goal:** Bridge the gap between isolated component tests (Phases 2-4) and the full
rollout function (Phase 5). Run the agent with a real SGLang server and a real sandbox
but limit to 3 turns, so you can isolate interaction bugs before running the full pipeline.

This catches issues that only appear when both components are live:
- Tool execution output too large for context window
- Sandbox state changes confuse the model's next tool call
- `asyncio.wait_for` timeouts interact with real sandbox latency

```python
# examples/mlebench/tests/test_real_integration.py
"""Test with real RolloutClient AND real sandbox, limited turns."""

import asyncio
from transformers import AutoTokenizer
from rllm.experimental.fully_async.client import RolloutClient

async def test_real_integration():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    client = RolloutClient(
        router_url="http://localhost:30001",
        tokenizer=tokenizer,
    )

    from examples.mlebench.train import _create_sandbox, AGENT_CONFIG
    AGENT_CONFIG.update({
        "mle_bench_data_dir": "/path/to/data",
        "manager_uri": "http://localhost:8080",
        "superimage_directory": "/path/to/superimage",
        "superimage_version": "v1",
        "superimage_overlay": "/path/to/overlay",
    })

    sandbox = await asyncio.to_thread(
        _create_sandbox, {"instance_id": "digit-recognizer"}
    )

    try:
        from examples.mlebench.agent import MLEBenchAgent

        agent = MLEBenchAgent(
            client=client,
            sandbox=sandbox,
            max_turns=3,  # Limited turns to keep test fast
            session_timeout=60,
            task_id="digit-recognizer",
            mle_bench_data_dir=AGENT_CONFIG["mle_bench_data_dir"],
        )

        from mle_agent.prompts_csv import INSTANCE_PROMPT, SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=1, timeout_unit="minutes",
            context_size=131072, rollout_timeout_hrs=1, max_turns=3,
        )
        instance_prompt = INSTANCE_PROMPT.format(task_description="Classify handwritten digits...")

        result = await agent.run([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ])

        print(f"Turns: {result['metrics']['num_turns']}")
        print(f"Termination: {result['metrics']['termination_reason']}")
        print(f"Sequences: {len(result['trajectory'].sequences)}")

        # Verify real data flowed through both components
        assert len(result["trajectory"].sequences) > 0
        for i, seq in enumerate(result["trajectory"].sequences):
            assert len(seq.response_ids) > 0, f"Seq {i} has no response tokens"
            assert all(lp < 0 for lp in seq.response_logprobs), f"Seq {i} has invalid logprobs"
            print(f"  Seq {i}: prompt={len(seq.prompt_ids)} response={len(seq.response_ids)}")

        print("PASSED: test_real_integration")
    finally:
        await asyncio.to_thread(sandbox.close)

if __name__ == "__main__":
    asyncio.run(test_real_integration())
```

**Prerequisites:** Both SGLang server AND AgentBox manager running.

**What to verify:**
- [ ] Model tool calls execute successfully in sandbox
- [ ] Tool output is appended to messages and influences next LLM call
- [ ] No timeout issues with real sandbox latency
- [ ] Trajectory sequences have real logprobs and correct token counts

---

### Phase 5: Full Rollout Function (Day 6)

**Goal:** Run `rollout_fn` end-to-end and validate trajectory structure.

```python
# examples/mlebench/tests/test_rollout_fn.py
"""Test complete rollout function."""

import asyncio

async def test_rollout_fn():
    from examples.mlebench.train import rollout_fn, AGENT_CONFIG, SANDBOX_SEMAPHORE
    import asyncio
    
    # Configure
    AGENT_CONFIG.update({...})
    SANDBOX_SEMAPHORE = asyncio.Semaphore(2)
    
    # Use real or mock client
    from rllm.experimental.fully_async.client import RolloutClient
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    client = RolloutClient(router_url="http://localhost:30001", tokenizer=tokenizer)
    
    # Run rollout
    trajectory = await rollout_fn(
        client, tokenizer,
        instance_id="digit-recognizer",
        task_description="Classify handwritten digits...",
    )
    
    # Validate
    assert trajectory.reward is not None
    assert len(trajectory.sequences) > 0
    
    merged = trajectory.merge()
    for seq in merged:
        assert len(seq.prompt_ids) > 0
        assert len(seq.response_ids) == len(seq.response_logprobs)
        assert len(seq.response_ids) == len(seq.response_masks)
    
    print(f"Reward: {trajectory.reward}")
    print(f"Metadata: {trajectory.metadata}")
    print("✅ Phase 5 PASSED: Full rollout works")

if __name__ == "__main__":
    asyncio.run(test_rollout_fn())
```

---

### Phase 6: Training Integration (Day 7+)

**Goal:** Run training for 1 step to verify data flows correctly.

```bash
# Minimal training run
python -m examples.mlebench.train \
    trainer.total_epochs=1 \
    async_training.required_samples=2 \
    async_training.trigger_parameter_sync_step=1 \
    +mlebench.max_turns=5 \
    +mlebench.max_concurrent_sandboxes=2 \
    data.train_dataset_name=mlebench \
    data.train_split=train
```

**What to verify:**
- [ ] Trainer starts without import errors
- [ ] First trajectory reaches MessageQueue
- [ ] Trainer pulls trajectory and computes loss (no NaN)
- [ ] Weight sync happens (check logs for `[ParameterSynchronizer]`)
- [ ] Second trajectory uses updated weights
- [ ] All rollout failures produce `None` results that are filtered by `TrajectoryGroup` (not crashes)

---

## Testing Checklist

| Phase | Component | Dependencies | Test File | Can Parallelize | Status |
|-------|-----------|--------------|-----------|-----------------|--------|
| 1 | Agent + Mocks (multi-turn, error paths) | None | `test_mock.py` | — | |
| 2 | Real RolloutClient + tool parsing | SGLang server | `test_real_client.py` | — | |
| 3 | Sandbox lifecycle (variable hold times) | AgentBox manager | `test_sandbox_concurrency.py` | With Phase 4 | |
| 4 | Reward computation | mlebench data | `test_reward.py` | With Phase 3 | |
| 4.5 | Real LLM + Real Sandbox (limited turns) | SGLang + AgentBox | `test_real_integration.py` | — | |
| 5 | Full rollout_fn | All above | `test_rollout_fn.py` | — | |
| 6 | Training integration | verl, Ray | CLI command | — | |

**Key principle:** Each phase isolates one component. If Phase N fails, the bug is in that component, not earlier ones. Phases 3 and 4 are independent and should run in parallel.
