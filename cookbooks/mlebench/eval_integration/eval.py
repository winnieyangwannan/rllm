#!/usr/bin/env python3
"""MLE-bench Evaluation Script with YAML Config Support.

This script runs MLE-bench evaluations using configuration files.
Based on test_step7_end_to_end.py but with externalized config.

For multi-node evaluation, see eval_ray.py which distributes rollouts
across a Ray cluster.

Usage:
    cd /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration
    conda activate rllm

    # Basic usage with config file
    python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

    # Override samples via CLI
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 4

    # Run multiple tasks
    python launch.py --config configs/gpt5.yaml --nodes 4 --samples 64
    
    # Run all tasks from JSONL directory
    python eval.py --config configs/gpt5.yaml --all-tasks

    # Custom output directory
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --output-dir /path/to/output

    # Skip saving trajectories
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --no-save

    # Or run from anywhere with absolute paths:
    python /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration/eval.py \
        --config /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration/configs/gpt5_test.yaml \
        --task mlsp-2013-birds
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# Add mle_agent to path
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")

# Import rLLM types for trajectory saving
from rllm.types import Episode, Trajectory


@dataclass
class RolloutOutput:
    """Output from run_agent — the pure rollout result before evaluation."""

    steps: list
    messages: list
    pred_solution: str | None
    sandbox: Any  # AgentBoxSandbox instance, kept open for evaluation
    task_id: str
    sample_idx: int
    duration: float


@dataclass
class EvalResult:
    """Result from end-to-end evaluation."""

    task_id: str
    sample_idx: int
    success: bool
    percentile: float
    score: float | None
    num_steps: int
    duration: float
    error: str | None = None
    pred_solution: str | None = None
    # Trajectory data for saving
    steps: list | None = None
    messages: list | None = None

    # NEW: Token metrics (from OpenAI API usage)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # NEW: Duration breakdown
    rollout_duration: float = 0.0  # Agent interaction time
    eval_duration: float = 0.0  # Evaluation/scoring time (CODE mode: includes train.py execution)

    # NEW: Termination info
    termination_reason: str | None = None  # "submit", "max_turns", "rollout_timeout", "format_error", "model_call_error", "error"

    # NEW: Outcome flags (matching amaia-collab OutcomeMetrics)
    outcomes: dict | None = None


def load_config(config_path: str) -> OmegaConf:
    """Load and merge YAML config files.

    Supports OmegaConf 'defaults' for inheritance:
    - defaults: [base] will load base.yaml first, then merge current config
    """
    config_path = Path(config_path)
    config_dir = config_path.parent

    # Load the main config
    cfg = OmegaConf.load(config_path)

    # Handle defaults (config inheritance)
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        merged_cfg = OmegaConf.create()

        for default in defaults:
            if isinstance(default, str):
                default_path = config_dir / f"{default}.yaml"
                if default_path.exists():
                    default_cfg = OmegaConf.load(default_path)
                    merged_cfg = OmegaConf.merge(merged_cfg, default_cfg)

        # Merge the current config on top of defaults
        cfg = OmegaConf.merge(merged_cfg, cfg)

    # Resolve environment variables and interpolations
    OmegaConf.resolve(cfg)

    return cfg


def load_task_from_jsonl(task_id: str, task_path: str) -> dict[str, Any]:
    """Load a single task instance from JSONL file.

    Args:
        task_id: The instance_id of the task to load
        task_path: Path to JSONL file containing all tasks (one task per line)

    Returns:
        Task dict with instance_id, task_description, etc.
    """
    jsonl_path = Path(task_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("instance_id") == task_id:
                return data

    raise ValueError(f"Task {task_id} not found in {jsonl_path}")


def load_tasks_from_jsonl(task_ids: list[str], task_path: str) -> dict[str, dict[str, Any]]:
    """Load multiple tasks from JSONL file in a single pass.

    More efficient than calling load_task_from_jsonl multiple times,
    as it reads the file only once.

    Args:
        task_ids: List of instance_ids to load
        task_path: Path to JSONL file containing all tasks

    Returns:
        Dict mapping instance_id -> task dict
    """
    jsonl_path = Path(task_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")

    wanted = set(task_ids)
    found: dict[str, dict[str, Any]] = {}

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if instance_id in wanted:
                found[instance_id] = data
                if len(found) == len(wanted):
                    break  # Found all, stop early

    # Check for missing tasks
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Tasks not found in {jsonl_path}: {missing}")

    return found


def list_available_tasks(task_path: str) -> list[str]:
    """List all available task IDs from JSONL file.

    Args:
        task_path: Path to JSONL file containing all tasks

    Returns:
        Sorted list of instance_ids from the file
    """
    jsonl_path = Path(task_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")

    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if instance_id:
                tasks.append(instance_id)
    return sorted(tasks)


def run_single_rollout(
    task_data: dict[str, Any],
    sample_idx: int,
    cfg: OmegaConf,
) -> EvalResult:
    """Run a single agent rollout and evaluate.

    Args:
        task_data: Task dictionary with instance_id, task_description, etc.
        sample_idx: Index of this sample (0-based).
        cfg: Full config object.
    """
    import openai
    from agentbox import ContainerConfig
    from mle_agent.agent import _run_agent_loop
    from mle_agent.evaluator import MLEEvaluator

    # Import prompts based on submit_file mode (csv or code)
    if cfg.agent.submit_file == "csv":
        from mle_agent.prompts_csv import INSTANCE_PROMPT, SYSTEM_PROMPT
    else:  # code mode
        from mle_agent.prompts_code import INSTANCE_PROMPT, SYSTEM_PROMPT

    from rllm.sdk.sandbox.backends.agentbox_backend import AgentBoxSandbox

    task_id = task_data["instance_id"]
    task_description = task_data.get("task_description", "")

    print(f"\n{'=' * 60}")
    print(f"ROLLOUT {sample_idx + 1}: {task_id}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Build data path for this task
    data_path = f"{cfg.data.mle_bench_data_dir}/{task_id}/prepared/public"
    print(f"Data path: {data_path}")

    # Build container config with data mount
    container_config = ContainerConfig(
        superimage_directory=cfg.sandbox.superimage_directory,
        superimage_version=cfg.sandbox.superimage_version,
        container_runtime="apptainer",
        read_only_overlays=[cfg.sandbox.superimage_overlay],
        read_only_binds={data_path: "/root/data"},
        working_dir="/workspace",
        env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
    )

    # Create sandbox with container config
    sandbox = AgentBoxSandbox(
        name=f"eval-{task_id}-{sample_idx}",
        manager_uri=cfg.sandbox.manager_uri,
        container_config=container_config,
    )
    print(f"✓ Created sandbox with data mount: {data_path} -> /root/data")

    try:
        # Set up workspace
        sandbox.exec("mkdir -p /workspace")

        # Gather data info from container
        data_info_cmd = """cd /root/data && \
echo "=== DATA STRUCTURE ===" && ls -sh && \
echo -e "\\n=== CSV ROW COUNTS ===" && wc -l *.csv 2>/dev/null && \
echo -e "\\n=== SAMPLE SUBMISSION FORMAT ===" && head -3 sample_submission.csv 2>/dev/null"""
        _data_info = sandbox.exec(data_info_cmd, timeout=30)  # noqa: F841
        print("✓ Gathered data info")

        # Build prompts based on submit_file mode
        if cfg.agent.submit_file == "csv":
            system_prompt = SYSTEM_PROMPT.format(
                timeout_min=int(cfg.agent.session_timeout / 60),
                timeout_unit="minutes",
                context_size=cfg.agent.context_size,
                rollout_timeout_hrs=int(cfg.agent.rollout_timeout / 3600),
                max_turns=cfg.agent.max_turns,
            )
        else:  # code mode
            system_prompt = SYSTEM_PROMPT.format(
                timeout_min=int(cfg.agent.session_timeout / 60),
                context_size=cfg.agent.context_size,
                eval_timeout_hrs=int(cfg.agent.rollout_timeout / 3600),
                max_turns=cfg.agent.max_turns,
            )

        instance_prompt = INSTANCE_PROMPT.format(
            task_description=task_description,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        # Create OpenAI client
        client = openai.AzureOpenAI(
            azure_endpoint=cfg.model.azure_endpoint,
            api_key=cfg.model.api_key,
            api_version=cfg.model.api_version,
        )
        print(f"✓ Created AzureOpenAI client (model: {cfg.model.name})")

        # Run agent loop
        print(f"Starting agent loop (max_turns={cfg.agent.max_turns}, timeout={cfg.agent.rollout_timeout}s, context_size={cfg.agent.context_size})...")
        print(f"Tools: bash, edit, create, submit ({cfg.agent.submit_file} mode), check_submission_validity")
        reasoning_effort = getattr(cfg.model, "reasoning_effort", None)
        if reasoning_effort:
            print(f"Reasoning effort: {reasoning_effort}")
        steps, final_messages, pred_solution, rollout_metrics = _run_agent_loop(
            client=client,
            model=cfg.model.name,
            messages=messages,
            sandbox=sandbox,
            max_turns=cfg.agent.max_turns,
            session_timeout=cfg.agent.session_timeout,
            rollout_timeout=cfg.agent.rollout_timeout,
            context_size=cfg.agent.context_size,
            temperature=cfg.agent.temperature,
            check_submission_validity=cfg.agent.check_submission_validity,
            task_id=task_id,
            mle_bench_data_dir=cfg.data.mle_bench_data_dir,
            submit_file=cfg.agent.submit_file,
            reasoning_effort=reasoning_effort,
        )

        rollout_end_time = time.time()
        rollout_duration = rollout_metrics.get("rollout_duration", rollout_end_time - start_time)
        print(f"✓ Agent completed with {len(steps)} steps in {rollout_duration:.1f}s")
        print(f"  Termination: {rollout_metrics.get('termination_reason', 'unknown')}")
        print(f"  Tokens: {rollout_metrics.get('prompt_tokens', 0)} prompt + {rollout_metrics.get('completion_tokens', 0)} completion")

        # Log steps summary
        for i, step in enumerate(steps):
            tool = step.input.get("tool", "?") if isinstance(step.input, dict) else "?"
            print(f"  Step {i + 1}: {tool}")

        if pred_solution:
            print(f"✓ Solution submitted ({len(pred_solution)} chars)")
        else:
            print("⚠ No solution submitted")

        # Run evaluation
        print("\nRunning evaluation...")
        eval_start_time = time.time()
        evaluator = MLEEvaluator(
            mle_bench_data_dir=cfg.data.mle_bench_data_dir,
            eval_timeout=cfg.agent.get("eval_timeout", 32400),  # 9 hours default for code mode
            submit_file=cfg.agent.submit_file,
        )

        # Create mock Episode-like object for evaluator
        class MockEpisode:
            def __init__(self, pred_solution, sandbox):
                self.artifacts = {
                    "_sandbox": sandbox,
                    "pred_solution": pred_solution,
                }

        task = {"task_id": task_id, "instance_id": task_id}
        episode = MockEpisode(pred_solution, sandbox)

        eval_output = evaluator.evaluate(task, episode)
        eval_duration = time.time() - eval_start_time

        # Convert signals list to dict for easier access
        signals_dict = {s.name: s.value for s in eval_output.signals}

        print("✓ Evaluation complete:")
        print(f"  Percentile: {eval_output.reward:.4f}")
        print(f"  Signals: {signals_dict}")
        print(f"  Eval duration: {eval_duration:.1f}s")

        # Build outcomes dict (matching amaia-collab OutcomeMetrics)
        termination_reason = rollout_metrics.get("termination_reason", "unknown")
        outcomes = {
            # Success/failure flags
            "pass": eval_output.reward > 0,
            "valid_submission": signals_dict.get("valid_submission", 0.0) > 0,
            "pred_solution_provided": pred_solution is not None,
            "submission_csv_provided": signals_dict.get("submission_csv_provided", 0.0) > 0,
            # Termination status
            "max_turns_reached": termination_reason == "max_turns",
            "rollout_timeout": termination_reason == "rollout_timeout",
            "context_exceeded": termination_reason == "context_exceeded",
            "model_call_error": termination_reason == "model_call_error",
            "parse_error": termination_reason == "format_error",
            # Eval timeout (code mode: train.py exceeded eval_timeout)
            "eval_timeout": cfg.agent.submit_file == "code" and eval_duration > cfg.agent.eval_timeout,
            # Outcome details
            "eval_outcome": "pass" if eval_output.reward > 0 else "fail",
            "eval_error_message": "",
            "eval_error_output": "",
        }

        total_duration = time.time() - start_time

        return EvalResult(
            task_id=task_id,
            sample_idx=sample_idx,
            success=eval_output.reward > 0,
            percentile=eval_output.reward,
            score=signals_dict.get("raw_score"),
            num_steps=len(steps),
            duration=total_duration,
            pred_solution=pred_solution,
            steps=steps,
            messages=final_messages,
            # NEW fields
            prompt_tokens=rollout_metrics.get("prompt_tokens", 0),
            completion_tokens=rollout_metrics.get("completion_tokens", 0),
            rollout_duration=rollout_duration,
            eval_duration=eval_duration,
            termination_reason=termination_reason,
            outcomes=outcomes,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_duration = time.time() - start_time
        return EvalResult(
            task_id=task_id,
            sample_idx=sample_idx,
            success=False,
            percentile=0.0,
            score=None,
            num_steps=0,
            duration=error_duration,
            error=str(e),
            steps=None,
            messages=None,
            # NEW fields for error case
            prompt_tokens=0,
            completion_tokens=0,
            rollout_duration=error_duration,
            eval_duration=0.0,
            termination_reason="error",
            outcomes={
                "pass": False,
                "valid_submission": False,
                "pred_solution_provided": False,
                "submission_csv_provided": False,
                "max_turns_reached": False,
                "rollout_timeout": False,
                "context_exceeded": False,
                "model_call_error": False,
                "parse_error": False,
                "eval_timeout": False,
                "eval_outcome": "exception",
                "eval_error_message": str(e),
                "eval_error_output": traceback.format_exc()[:4096],
            },
        )

    finally:
        sandbox.close()
        print("✓ Sandbox closed")


def build_trajectory_dict(result: EvalResult, task_data: dict, cfg: OmegaConf) -> dict:
    """Build trajectory/episode as a dictionary for JSONL output.

    Returns:
        Dictionary representation of the episode.
    """
    # Build Trajectory and Episode objects
    trajectory = Trajectory(
        name="mle_agent",
        task=task_data,
        steps=result.steps or [],
        output=result.pred_solution,
        reward=result.percentile,
        signals={
            "score": result.score or 0.0,
            "success": 1.0 if result.success else 0.0,
        },
    )

    episode = Episode(
        id=f"{result.task_id}:{result.sample_idx}",
        task=task_data,
        termination_reason=result.termination_reason,
        is_correct=result.success,
        trajectories=[trajectory],
        artifacts={
            "pred_solution": result.pred_solution,
            "messages": result.messages,
            "config": OmegaConf.to_container(cfg),
        },
        metrics={
            # Core performance metrics
            "percentile": result.percentile,
            "score": result.score,
            "num_steps": result.num_steps,
            "duration": result.duration,
            # Token metrics (NEW)
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
            # Duration breakdown (NEW)
            "rollout_duration": result.rollout_duration,
            "eval_duration": result.eval_duration,
        },
    )

    # Convert to dict and add outcomes at top level
    episode_dict = episode.model_dump()
    episode_dict["outcomes"] = result.outcomes or {}

    return episode_dict


def save_trajectory(result: EvalResult, task_data: dict, output_dir: Path, cfg: OmegaConf) -> str:
    """Save trajectory to individual JSON file (legacy format).

    Returns:
        Path to saved file.
    """
    episode_dict = build_trajectory_dict(result, task_data, cfg)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{result.task_id}_{result.sample_idx}_{timestamp}.json"
    output_path.write_text(json.dumps(episode_dict, indent=2))

    return str(output_path)


def trajectory_writer(
    write_queue: queue.Queue,
    output_path: Path,
    num_expected: int,
    exc_queue: queue.Queue,
) -> None:
    """Dedicated thread for writing trajectories to JSONL file.

    Consumes from write_queue and writes each item as a line in JSONL.
    Expects exactly num_expected items, then a single None as shutdown signal.

    Args:
        write_queue: Queue containing (result, task_data, cfg) tuples, then None
        output_path: Path to output JSONL file
        num_expected: Number of results expected
        exc_queue: Queue for reporting exceptions
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written_count = 0
        received_count = 0

        with open(output_path, "w") as f:
            while received_count < num_expected:
                item = write_queue.get()
                received_count += 1

                if item is None:
                    # Skip failed rollouts (they send None as placeholder)
                    continue

                # Unpack and write
                result, task_data, cfg = item
                try:
                    episode_dict = build_trajectory_dict(result, task_data, cfg)
                    f.write(json.dumps(episode_dict) + "\n")
                    f.flush()  # Flush after each write for crash safety
                    written_count += 1
                    print(f"✓ Writer: saved rollout {result.sample_idx} ({received_count}/{num_expected})")
                except Exception as e:
                    print(f"⚠ Writer: failed to save rollout {result.sample_idx}: {e}")

        print(f"✓ Writer thread: wrote {written_count} trajectories to {output_path}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        exc_queue.put(e)


def run_task_eval(
    task_id: str,
    cfg: OmegaConf,
    num_samples: int | None = None,
    output_dir: str | None = None,
) -> list[EvalResult]:
    """Run evaluation on a single task with N samples.

    Uses a dedicated writer thread for thread-safe JSONL output.
    """

    # Use CLI overrides or config defaults
    num_samples = num_samples or cfg.eval.samples_per_prompt
    output_dir = output_dir or cfg.eval.get("output_dir")
    parallel = cfg.eval.get("parallel", True)
    max_workers = cfg.eval.get("max_workers") or num_samples  # Default to num_samples if not set

    print("\n" + "=" * 70)
    print(f"TASK EVAL: {task_id}")
    print("=" * 70)
    print("Config:")
    print(f"  model: {cfg.model.name}")
    print(f"  samples_per_prompt: {num_samples}")
    print(f"  max_workers: {max_workers}")
    print(f"  rollout_timeout: {cfg.agent.rollout_timeout}s ({cfg.agent.rollout_timeout / 60:.0f} min)")
    print(f"  session_timeout: {cfg.agent.session_timeout}s ({cfg.agent.session_timeout / 60:.0f} min)")
    print(f"  max_turns: {cfg.agent.max_turns}")
    print(f"  manager_uri: {cfg.sandbox.manager_uri}")
    print(f"  output_dir: {output_dir or 'None (no saving)'}")
    print(f"  submit_file: {cfg.agent.submit_file}")
    print(f"  parallel: {parallel}")

    # Load task
    print("\nLoading task from JSONL...")
    task_data = load_task_from_jsonl(task_id, cfg.data.task_path)
    print(f"✓ Loaded task: {task_id}")
    print(f"  Difficulty: {task_data.get('difficulty', 'unknown')}")
    print(f"  Description length: {len(task_data.get('task_description', ''))} chars")

    # Set up queues and writer thread if saving is enabled
    write_queue = None
    writer_thread = None
    exc_queue = queue.Queue()

    if output_dir:
        output_path = Path(output_dir) / f"{task_id}.jsonl"
        write_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=trajectory_writer,
            kwargs={
                "write_queue": write_queue,
                "output_path": output_path,
                "num_expected": num_samples,
                "exc_queue": exc_queue,
            },
            daemon=True,
        )
        writer_thread.start()
        print(f"✓ Started writer thread for {output_path}")

    # Run rollouts
    results = []

    if parallel and num_samples > 1:
        print(f"\nRunning {num_samples} rollouts in parallel (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_rollout, task_data, i, cfg): i for i in range(num_samples)}
            for future in as_completed(futures):
                sample_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    # Queue trajectory for writing if enabled
                    if write_queue:
                        if result.steps:
                            write_queue.put((result, task_data, cfg))
                        else:
                            write_queue.put(None)  # Placeholder for failed/empty rollout
                except Exception as e:
                    print(f"⚠ Rollout {sample_idx} failed with exception: {e}")
                    results.append(
                        EvalResult(
                            task_id=task_id,
                            sample_idx=sample_idx,
                            success=False,
                            percentile=0.0,
                            score=None,
                            num_steps=0,
                            duration=0.0,
                            error=str(e),
                        )
                    )
                    # Send placeholder for failed rollout
                    if write_queue:
                        write_queue.put(None)
        # Sort results by sample_idx for consistent ordering
        results.sort(key=lambda r: r.sample_idx)
    else:
        for i in range(num_samples):
            result = run_single_rollout(task_data, i, cfg)
            results.append(result)

            # Queue trajectory for writing if enabled
            if write_queue:
                if result.steps:
                    write_queue.put((result, task_data, cfg))
                else:
                    write_queue.put(None)

    # Wait for writer thread to finish
    if writer_thread:
        # Send remaining kill signals if some rollouts had steps (and thus sent data, not None)
        # Actually we need to count how many data items vs None we sent
        # Simpler: just wait with timeout
        writer_thread.join(timeout=60)
        if writer_thread.is_alive():
            print("⚠ Writer thread did not finish in time")

        # Check for exceptions
        try:
            exc = exc_queue.get_nowait()
            print(f"⚠ Writer thread exception: {exc}")
        except queue.Empty:
            pass

    return results


def print_summary(all_results: dict[str, list[EvalResult]]):
    """Print summary of all task evaluations."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_samples = 0
    total_successful = 0
    all_percentiles = []

    for task_id, results in all_results.items():
        successful = [r for r in results if r.success]
        percentiles = [r.percentile for r in results if r.percentile > 0]

        print(f"\n{task_id}:")
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"  Sample {r.sample_idx + 1}: {status} percentile={r.percentile:.4f}, steps={r.num_steps}, duration={r.duration:.1f}s")
            if r.error:
                print(f"    Error: {r.error[:100]}")

        print(f"  Results: {len(successful)}/{len(results)} successful")
        if percentiles:
            print(f"  Mean percentile: {sum(percentiles) / len(percentiles):.4f}")
            print(f"  Max percentile: {max(percentiles):.4f}")

        total_samples += len(results)
        total_successful += len(successful)
        all_percentiles.extend(percentiles)

    print("\n" + "-" * 70)
    print("OVERALL:")
    print(f"  Total: {total_successful}/{total_samples} successful across {len(all_results)} tasks")
    if all_percentiles:
        print(f"  Mean percentile: {sum(all_percentiles) / len(all_percentiles):.4f}")
        print(f"  Max percentile: {max(all_percentiles):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="MLE-bench Evaluation with YAML Config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single task with test config
    python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

    # Multiple tasks
    python eval.py --config configs/gpt5.yaml --tasks mlsp-2013-birds,spooky-author-identification

    # Override samples
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 4

    # Custom output directory
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --output-dir /path/to/output
        """,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task", type=str, help="Single task ID to evaluate")
    parser.add_argument("--tasks", type=str, help="Comma-separated list of task IDs")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks from JSONL directory")
    parser.add_argument("--samples", type=int, help="Override samples_per_prompt from config")
    parser.add_argument("--output-dir", type=str, help="Override output directory from config")
    parser.add_argument("--no-save", action="store_true", help="Skip saving trajectories")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)
    print("✓ Config loaded")

    # List tasks mode
    if args.list_tasks:
        tasks = list_available_tasks(cfg.data.task_path)
        print(f"\nAvailable tasks ({len(tasks)}):")
        for task in tasks:
            print(f"  - {task}")
        return

    # Determine tasks to run
    task_ids = []
    if args.task:
        task_ids = [args.task]
    elif args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    elif args.all_tasks:
        task_ids = list_available_tasks(cfg.data.task_path)
    else:
        parser.error("Must specify --task, --tasks, or --all-tasks")

    # Determine output directory
    output_dir = None if args.no_save else (args.output_dir or cfg.eval.get("output_dir"))

    # Run evaluations
    all_results = {}
    for task_id in task_ids:
        try:
            results = run_task_eval(
                task_id=task_id,
                cfg=cfg,
                num_samples=args.samples,
                output_dir=output_dir,
            )
            all_results[task_id] = results
        except Exception as e:
            print(f"\n⚠ Failed to evaluate task {task_id}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    if all_results:
        print_summary(all_results)

    # Exit with success if any rollout succeeded
    any_success = any(r.success for results in all_results.values() for r in results)
    sys.exit(0 if any_success else 1)


if __name__ == "__main__":
    main()
