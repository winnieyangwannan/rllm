#!/usr/bin/env python3
"""Ray-Based Multi-Node MLE-bench Evaluation Script.

Distributes rollouts across a Ray cluster for parallel execution.
Falls back to local Ray cluster when no external cluster is available.

KEY FEATURE: All tasks run in parallel, not sequentially!
When running multiple tasks, all (task, sample) pairs are submitted to Ray
at once. Ray load-balances across all of them automatically, which is better
when tasks have different difficulty levels.

Usage:
    cd /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration
    conda activate rllm

    # Single-node mode (starts local Ray cluster automatically)
    python eval_ray.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 4

    # Multi-node mode (connect to existing Ray cluster)
    RAY_ADDRESS=auto python eval_ray.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 192

    # Multi-task parallel - all tasks interleaved for optimal load balancing
    python eval_ray.py --config configs/gpt5.yaml --tasks "task1,task2,task3" --samples 64

    # Explicit Ray head address
    RAY_ADDRESS=192.168.1.100:6379 python eval_ray.py --config configs/gpt5.yaml --task mlsp-2013-birds

    # Launch via SLURM (starts Ray cluster across nodes)
    python launch.py --config configs/gpt5.yaml --task mlsp-2013-birds --nodes 3 --samples 192

Architecture:
    - Each rollout is a stateless @ray.remote task
    - Ray schedules tasks across available nodes automatically
    - Multi-task: all (task, sample) pairs in single pool → Ray load-balances
    - Results collected via ray.wait() for real-time progress
    - Trajectories saved incrementally as rollouts complete
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import ray
from omegaconf import OmegaConf

# Add mle_agent to path
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")

# Import from main eval.py (reuse everything)
from eval import (
    EvalResult,
    build_trajectory_dict,
    list_available_tasks,
    load_config,
    load_task_from_jsonl,
    load_tasks_from_jsonl,
    print_summary,
    run_single_rollout,
)

# Import Ray init utilities from rllm
from rllm.trainer.ray_init_utils import get_ray_init_settings

# Default max concurrent tasks per node (can be overridden via config)


def get_completed_samples_by_task(jsonl_path: Path) -> dict[str, set[int]]:
    """Read existing JSONL file and return dict of completed sample indices per task.

    Used for resume mode to skip already-completed samples.
    Parses task_id and sample_idx from episode id format: "{task_id}:{sample_idx}"

    Returns:
        dict mapping task_id -> set of completed sample indices
    """
    completed = {}  # task_id -> set of sample_idx
    if not jsonl_path.exists():
        return completed

    try:
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Parse episode id: "task_id:sample_idx"
                        episode_id = data.get("id", "")
                        if ":" in episode_id:
                            try:
                                parts = episode_id.rsplit(":", 1)
                                task_id = parts[0]
                                sample_idx = int(parts[1])
                                if task_id not in completed:
                                    completed[task_id] = set()
                                completed[task_id].add(sample_idx)
                            except (ValueError, IndexError):
                                continue
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Failed to read {jsonl_path}: {e}")

    return completed


# This is the PRIMARY parallelism control - explicit throttling via ray.wait()
# This is the per-node default; actual max_concurrent = per_node * num_nodes
DEFAULT_MAX_CONCURRENT_TASKS_PER_NODE = 64


def get_num_ray_nodes() -> int:
    """Get the number of active Ray nodes in the cluster."""
    try:
        nodes = ray.nodes()
        # Count only alive nodes
        alive_nodes = [n for n in nodes if n.get("Alive", False)]
        return max(1, len(alive_nodes))
    except Exception:
        return 1


@ray.remote(num_cpus=1)  # Minimal resource claim - parallelism controlled by max_concurrent_tasks
def run_rollout_task(task_data: dict, sample_idx: int, cfg_dict: dict) -> dict:
    """Stateless Ray task wrapping run_single_rollout.

    Args:
        task_data: Task dictionary (already picklable dict)
        sample_idx: Sample index for this rollout
        cfg_dict: Config as dict (OmegaConf isn't picklable)

    Returns:
        EvalResult as dict (for Ray serialization)
    """
    # Reconstruct OmegaConf inside the task
    cfg = OmegaConf.create(cfg_dict)

    # Run the existing rollout function
    result = run_single_rollout(task_data, sample_idx, cfg)

    # Convert to dict for Ray serialization
    return dataclasses.asdict(result)


def run_task_eval_ray(
    task_id: str,
    cfg: OmegaConf,
    num_samples: int | None = None,
    output_dir: str | None = None,
    max_concurrent: int | None = None,
) -> list[EvalResult]:
    """Run evaluation on a single task using Ray for parallelization.

    Dispatches rollouts with explicit throttling via max_concurrent_tasks.
    Uses ray.wait() to maintain a sliding window of concurrent tasks.
    """
    # Use CLI overrides or config defaults
    num_samples = num_samples or cfg.eval.samples_per_prompt
    output_dir = output_dir or cfg.eval.get("output_dir")

    # Scale max_concurrent with number of nodes (config specifies per-node value)
    if max_concurrent is None:
        per_node = cfg.get("ray", {}).get("max_concurrent_tasks_per_node", DEFAULT_MAX_CONCURRENT_TASKS_PER_NODE)
        num_nodes = get_num_ray_nodes()
        max_concurrent = per_node * num_nodes
        print(f"\nParallelism: {per_node} tasks/node × {num_nodes} nodes = {max_concurrent} total concurrent")

    print("\n" + "=" * 70)
    print(f"RAY TASK EVAL: {task_id}")
    print("=" * 70)
    print("Config:")
    print(f"  model: {cfg.model.name}")
    print(f"  samples: {num_samples}")
    print(f"  rollout_timeout: {cfg.agent.rollout_timeout}s ({cfg.agent.rollout_timeout / 60:.0f} min)")
    print(f"  session_timeout: {cfg.agent.session_timeout}s ({cfg.agent.session_timeout / 60:.0f} min)")
    print(f"  max_turns: {cfg.agent.max_turns}")
    print(f"  manager_uri: {cfg.sandbox.manager_uri}")
    print(f"  output_dir: {output_dir or 'None (no saving)'}")
    print(f"  submit_file: {cfg.agent.submit_file}")
    print(f"  max_concurrent_tasks: {max_concurrent}")

    # Print Ray cluster info
    print(f"\nRay cluster resources: {ray.cluster_resources()}")

    # Load task data
    print("\nLoading task from JSONL...")
    task_data = load_task_from_jsonl(task_id, cfg.data.task_path)
    print(f"✓ Loaded task: {task_id}")
    print(f"  Difficulty: {task_data.get('difficulty', 'unknown')}")
    print(f"  Description length: {len(task_data.get('task_description', ''))} chars")

    # Convert config to dict for Ray serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Submit tasks with explicit throttling via max_concurrent
    # This is the PRIMARY parallelism control - not resource-based
    start_time = time.time()

    # Prepare output directory if saving
    jsonl_path = None
    completed_samples = set()
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / "trajectories.jsonl"

        # Check for existing completed samples (resume mode)
        all_completed = get_completed_samples_by_task(jsonl_path)
        completed_samples = all_completed.get(task_id, set())
        if completed_samples:
            print(f"✓ Found {len(completed_samples)} completed samples for {task_id} in {jsonl_path}")
        print(f"✓ Will save results to {jsonl_path}")

    # Build list of samples to run (skip completed ones)
    samples_to_run = [i for i in range(num_samples) if i not in completed_samples]
    num_to_run = len(samples_to_run)

    if num_to_run == 0:
        print(f"\n✓ All {num_samples} samples already completed!")
        return []

    print(f"\nRunning {num_to_run} rollouts (max {max_concurrent} concurrent)...")
    if completed_samples:
        print(f"  Skipping {len(completed_samples)} already completed samples")

    results = []
    pending = []
    sample_queue = iter(samples_to_run)  # Iterator over samples to run
    completed = 0

    # Sliding window: maintain up to max_concurrent pending tasks
    while completed < num_to_run:
        # Submit new tasks up to max_concurrent limit
        while len(pending) < max_concurrent:
            try:
                next_sample = next(sample_queue)
                future = run_rollout_task.remote(task_data, next_sample, cfg_dict)
                pending.append((next_sample, future))
            except StopIteration:
                break

        if not pending:
            break

        # Wait for at least one task to complete
        pending_futures = [f for _, f in pending]
        done, _ = ray.wait(pending_futures, num_returns=1, timeout=None)

        # Process completed tasks
        done_set = set(done)
        new_pending = []
        for sample_idx, future in pending:
            if future in done_set:
                try:
                    result_dict = ray.get(future)
                    result = EvalResult(**result_dict)
                    results.append(result)
                    completed += 1

                    # Progress update
                    status = "✓" if result.success else "✗"
                    elapsed = time.time() - start_time
                    total_done = completed + len(completed_samples)
                    print(
                        f"[{completed}/{num_to_run}] ({total_done}/{num_samples} total) {status} sample={result.sample_idx} "
                        f"percentile={result.percentile:.4f} steps={result.num_steps} "
                        f"duration={result.duration:.1f}s elapsed={elapsed:.1f}s"
                    )

                    # Save trajectory incrementally
                    if jsonl_path and result.steps:
                        episode_dict = build_trajectory_dict(result, task_data, cfg)
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(episode_dict) + "\n")

                except Exception as e:
                    completed += 1
                    print(f"[{completed}/{num_to_run}] ✗ Task {sample_idx} failed: {e}")
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
            else:
                new_pending.append((sample_idx, future))
        pending = new_pending

    # Sort by sample_idx for consistent ordering
    results.sort(key=lambda r: r.sample_idx)

    total_time = time.time() - start_time
    if completed_samples:
        print(f"\n✓ Completed {num_to_run} new rollouts in {total_time:.1f}s ({len(completed_samples)} previously completed)")
    else:
        print(f"\n✓ All {num_to_run} rollouts completed in {total_time:.1f}s")

    return results


def run_all_tasks_parallel(
    task_ids: list[str],
    cfg: OmegaConf,
    num_samples: int,
    output_dir: str | None = None,
    max_concurrent: int | None = None,
) -> dict[str, list[EvalResult]]:
    """Run evaluation on ALL tasks in parallel using Ray.

    Unlike sequential task processing, this submits all (task, sample) pairs
    to Ray at once. Ray load-balances across all tasks automatically, which
    is better when tasks have different difficulty levels.

    Args:
        task_ids: List of task IDs to evaluate
        cfg: Configuration object
        num_samples: Number of samples per task
        output_dir: Directory for saving results
        max_concurrent: Max concurrent tasks (default from config)

    Returns:
        Dict mapping task_id -> list of EvalResults
    """
    # Scale max_concurrent with number of nodes (config specifies per-node value)
    if max_concurrent is None:
        per_node = cfg.get("ray", {}).get("max_concurrent_tasks_per_node", DEFAULT_MAX_CONCURRENT_TASKS_PER_NODE)
        num_nodes = get_num_ray_nodes()
        max_concurrent = per_node * num_nodes
        print(f"\nParallelism: {per_node} tasks/node × {num_nodes} nodes = {max_concurrent} total concurrent")
    total_rollouts = len(task_ids) * num_samples

    print("\n" + "=" * 70)
    print("RAY PARALLEL MULTI-TASK EVAL")
    print("=" * 70)
    print("Config:")
    print(f"  model: {cfg.model.name}")
    print(f"  tasks: {len(task_ids)}")
    print(f"  samples_per_task: {num_samples}")
    print(f"  total_rollouts: {total_rollouts}")
    print(f"  max_concurrent_tasks: {max_concurrent}")
    print(f"  rollout_timeout: {cfg.agent.rollout_timeout}s ({cfg.agent.rollout_timeout / 60:.0f} min)")
    print(f"  output_dir: {output_dir or 'None (no saving)'}")
    print(f"\nTasks: {', '.join(task_ids)}")
    print(f"\nRay cluster resources: {ray.cluster_resources()}")

    # Load all task data upfront (single file read)
    print("\nLoading task data...")
    task_data_map = load_tasks_from_jsonl(task_ids, cfg.data.task_path)
    for task_id in task_ids:
        print(f"  ✓ {task_id} (difficulty: {task_data_map[task_id].get('difficulty', 'unknown')})")

    # Convert config to dict for Ray serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Prepare output file and check for completed samples (resume mode)
    jsonl_path = None
    completed_samples = {task_id: set() for task_id in task_ids}
    total_previously_completed = 0
    all_results = {task_id: [] for task_id in task_ids}  # Initialize early for resume case
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / "trajectories.jsonl"

        # Check for existing completed samples (resume mode)
        all_completed = get_completed_samples_by_task(jsonl_path)
        for task_id in task_ids:
            completed_samples[task_id] = all_completed.get(task_id, set())
            if completed_samples[task_id]:
                print(f"  ✓ Found {len(completed_samples[task_id])} completed samples for {task_id}")
                total_previously_completed += len(completed_samples[task_id])
        print(f"\n✓ Will save results to {jsonl_path}")

    # Build work queue: all (task_id, sample_idx) pairs, skipping completed ones
    work_queue = [(task_id, sample_idx) for task_id in task_ids for sample_idx in range(num_samples) if sample_idx not in completed_samples[task_id]]
    num_to_run = len(work_queue)

    if num_to_run == 0:
        print(f"\n✓ All {total_rollouts} samples already completed!")
        return all_results

    print(f"\nSubmitting {num_to_run} rollouts (max {max_concurrent} concurrent)...")
    if total_previously_completed > 0:
        print(f"  Skipping {total_previously_completed} already completed samples")
    start_time = time.time()

    pending = []  # List of (task_id, sample_idx, future)
    next_work = 0
    completed = 0

    # Track per-task progress
    task_completed = {task_id: 0 for task_id in task_ids}

    # Sliding window: maintain up to max_concurrent pending tasks
    while completed < num_to_run:
        # Submit new tasks up to max_concurrent limit
        while next_work < num_to_run and len(pending) < max_concurrent:
            task_id, sample_idx = work_queue[next_work]
            task_data = task_data_map[task_id]
            future = run_rollout_task.remote(task_data, sample_idx, cfg_dict)
            pending.append((task_id, sample_idx, future))
            next_work += 1

        if not pending:
            break

        # Wait for at least one task to complete
        pending_futures = [f for _, _, f in pending]
        done, _ = ray.wait(pending_futures, num_returns=1, timeout=None)

        # Process completed tasks
        done_set = set(done)
        new_pending = []
        for task_id, sample_idx, future in pending:
            if future in done_set:
                try:
                    result_dict = ray.get(future)
                    result = EvalResult(**result_dict)
                    all_results[task_id].append(result)
                    completed += 1
                    task_completed[task_id] += 1

                    # Progress update with task info
                    status = "✓" if result.success else "✗"
                    elapsed = time.time() - start_time
                    task_prev = len(completed_samples[task_id])
                    task_progress = f"{task_completed[task_id] + task_prev}/{num_samples}"
                    total_done = completed + total_previously_completed
                    print(
                        f"[{completed}/{num_to_run}] ({total_done}/{total_rollouts} total) {status} {task_id}[{task_progress}] "
                        f"sample={result.sample_idx} percentile={result.percentile:.4f} "
                        f"steps={result.num_steps} duration={result.duration:.1f}s elapsed={elapsed:.1f}s"
                    )

                    # Save trajectory incrementally
                    if jsonl_path and result.steps:
                        task_data = task_data_map[task_id]
                        episode_dict = build_trajectory_dict(result, task_data, cfg)
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(episode_dict) + "\n")

                except Exception as e:
                    completed += 1
                    task_completed[task_id] += 1
                    print(f"[{completed}/{num_to_run}] ✗ {task_id} sample={sample_idx} failed: {e}")
                    all_results[task_id].append(
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
            else:
                new_pending.append((task_id, sample_idx, future))
        pending = new_pending

    # Sort results by sample_idx within each task
    for task_id in task_ids:
        all_results[task_id].sort(key=lambda r: r.sample_idx)

    total_time = time.time() - start_time
    if total_previously_completed > 0:
        print(f"\n✓ Completed {num_to_run} new rollouts in {total_time:.1f}s ({total_previously_completed} previously completed)")
    else:
        print(f"\n✓ All {total_rollouts} rollouts completed in {total_time:.1f}s")
    if num_to_run > 0:
        print(f"  Average: {total_time / num_to_run:.1f}s per rollout")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Ray-Based Multi-Node MLE-bench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-node (starts local Ray cluster)
    python eval_ray.py --config configs/gpt5_test.yaml --task mlsp-2013-birds --samples 4

    # Multi-node (use existing Ray cluster)
    RAY_ADDRESS=auto python eval_ray.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 192

    # Multi-task parallel (all tasks run in parallel, not sequentially)
    python eval_ray.py --config configs/gpt5.yaml --tasks "task1,task2,task3" --samples 64

    # Launch via SLURM
    python launch.py --config configs/gpt5.yaml --task mlsp-2013-birds --nodes 3 --samples 192
        """,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task", type=str, help="Single task ID to evaluate")
    parser.add_argument("--tasks", type=str, help="Comma-separated list of task IDs (run in parallel)")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks from JSONL directory (in parallel)")
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

    # Determine task list
    if args.task:
        task_ids = [args.task]
    elif args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    elif args.all_tasks:
        task_ids = list_available_tasks(cfg.data.task_path)
        print(f"Running all {len(task_ids)} tasks in parallel")
    else:
        parser.error("Must specify --task, --tasks, or --all-tasks")

    # Get sample count
    num_samples = args.samples or cfg.eval.samples_per_prompt

    # Get output directory
    output_dir = None if args.no_save else (args.output_dir or cfg.eval.get("output_dir"))

    # Initialize Ray
    print("\nInitializing Ray...")
    ray_settings = get_ray_init_settings()
    ray.init(**ray_settings)
    print("✓ Ray initialized")
    print(f"  Address: {ray.get_runtime_context().gcs_address}")
    print(f"  Resources: {ray.cluster_resources()}")

    try:
        # Run ALL tasks in parallel (not sequentially)
        # Ray load-balances across all (task, sample) pairs automatically
        all_results = run_all_tasks_parallel(
            task_ids=task_ids,
            cfg=cfg,
            num_samples=num_samples,
            output_dir=output_dir,
        )

        # Print summary
        print_summary(all_results)

    finally:
        # Cleanup Ray
        ray.shutdown()
        print("\n✓ Ray shutdown complete")


if __name__ == "__main__":
    main()
