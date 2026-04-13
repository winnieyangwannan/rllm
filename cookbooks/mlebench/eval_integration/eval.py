#!/usr/bin/env python3
"""MLE-bench Evaluation Script with YAML Config Support.

This script runs MLE-bench evaluations using configuration files.
Based on test_step7_end_to_end.py but with externalized config.

Usage:
    cd /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration
    conda activate rllm

    # Basic usage with config file
    python eval.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

    # Override samples via CLI
    python eval.py --config configs/gpt5.yaml --task mlsp-2013-birds --samples 4

    # Run multiple tasks
    python eval.py --config configs/gpt5.yaml --tasks mlsp-2013-birds,spooky-author-identification

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
import sys
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


def load_task_from_jsonl(task_id: str, task_jsonl_dir: str) -> dict[str, Any]:
    """Load task instance from JSONL file."""
    jsonl_path = Path(task_jsonl_dir) / f"{task_id}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("instance_id") == task_id:
                return data

    raise ValueError(f"Task {task_id} not found in {jsonl_path}")


def list_available_tasks(task_jsonl_dir: str) -> list[str]:
    """List all available task IDs from JSONL directory."""
    jsonl_dir = Path(task_jsonl_dir)
    tasks = []
    for jsonl_file in jsonl_dir.glob("*.jsonl"):
        task_id = jsonl_file.stem
        tasks.append(task_id)
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
    from mle_agent.prompts import INSTANCE_PROMPT, SYSTEM_PROMPT

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

        # Build prompts
        system_prompt = SYSTEM_PROMPT.format(
            timeout_min=int(cfg.agent.session_timeout / 60),
            context_size=cfg.agent.context_size,
            eval_timeout_hrs=int(cfg.agent.rollout_timeout / 3600),
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
        print(f"Starting agent loop (max_turns={cfg.agent.max_turns}, timeout={cfg.agent.rollout_timeout}s)...")
        print(f"Tools: bash, edit, create, submit ({cfg.agent.submit_file} mode), check_submission_validity")
        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=cfg.model.name,
            messages=messages,
            sandbox=sandbox,
            max_turns=cfg.agent.max_turns,
            session_timeout=cfg.agent.session_timeout,
            rollout_timeout=cfg.agent.rollout_timeout,
            temperature=cfg.agent.temperature,
            check_submission_validity=cfg.agent.check_submission_validity,
            task_id=task_id,
            mle_bench_data_dir=cfg.data.mle_bench_data_dir,
            submit_file=cfg.agent.submit_file,
        )

        duration = time.time() - start_time
        print(f"✓ Agent completed with {len(steps)} steps in {duration:.1f}s")

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
        evaluator = MLEEvaluator(
            mle_bench_data_dir=cfg.data.mle_bench_data_dir,
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

        # Convert signals list to dict for easier access
        signals_dict = {s.name: s.value for s in eval_output.signals}

        print("✓ Evaluation complete:")
        print(f"  Percentile: {eval_output.reward:.4f}")
        print(f"  Signals: {signals_dict}")

        return EvalResult(
            task_id=task_id,
            sample_idx=sample_idx,
            success=eval_output.reward > 0,
            percentile=eval_output.reward,
            score=signals_dict.get("score"),
            num_steps=len(steps),
            duration=duration,
            pred_solution=pred_solution,
            steps=steps,
            messages=final_messages,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return EvalResult(
            task_id=task_id,
            sample_idx=sample_idx,
            success=False,
            percentile=0.0,
            score=None,
            num_steps=0,
            duration=time.time() - start_time,
            error=str(e),
            steps=None,
            messages=None,
        )

    finally:
        sandbox.close()
        print("✓ Sandbox closed")


def save_trajectory(result: EvalResult, task_data: dict, output_dir: Path, cfg: OmegaConf) -> str:
    """Save trajectory to JSON file.

    Returns:
        Path to saved file.
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
        is_correct=result.success,
        trajectories=[trajectory],
        artifacts={
            "pred_solution": result.pred_solution,
            "messages": result.messages,
            "config": OmegaConf.to_container(cfg),
        },
        metrics={
            "percentile": result.percentile,
            "score": result.score,
            "num_steps": result.num_steps,
            "duration": result.duration,
        },
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{result.task_id}_{result.sample_idx}_{timestamp}.json"
    output_path.write_text(episode.model_dump_json(indent=2))

    return str(output_path)


def run_task_eval(
    task_id: str,
    cfg: OmegaConf,
    num_samples: int | None = None,
    output_dir: str | None = None,
) -> list[EvalResult]:
    """Run evaluation on a single task with N samples."""

    # Use CLI overrides or config defaults
    num_samples = num_samples or cfg.eval.samples_per_prompt
    output_dir = output_dir or cfg.eval.get("output_dir")
    parallel = cfg.eval.get("parallel", True)

    print("\n" + "=" * 70)
    print(f"TASK EVAL: {task_id}")
    print("=" * 70)
    print("Config:")
    print(f"  model: {cfg.model.name}")
    print(f"  samples_per_prompt: {num_samples}")
    print(f"  rollout_timeout: {cfg.agent.rollout_timeout}s ({cfg.agent.rollout_timeout / 60:.0f} min)")
    print(f"  session_timeout: {cfg.agent.session_timeout}s ({cfg.agent.session_timeout / 60:.0f} min)")
    print(f"  max_turns: {cfg.agent.max_turns}")
    print(f"  manager_uri: {cfg.sandbox.manager_uri}")
    print(f"  output_dir: {output_dir or 'None (no saving)'}")
    print(f"  submit_file: {cfg.agent.submit_file}")
    print(f"  parallel: {parallel}")

    # Load task
    print("\nLoading task from JSONL...")
    task_data = load_task_from_jsonl(task_id, cfg.data.task_jsonl_dir)
    print(f"✓ Loaded task: {task_id}")
    print(f"  Difficulty: {task_data.get('difficulty', 'unknown')}")
    print(f"  Description length: {len(task_data.get('task_description', ''))} chars")

    # Run rollouts
    results = []

    if parallel and num_samples > 1:
        print(f"\nRunning {num_samples} rollouts in parallel...")
        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            futures = {executor.submit(run_single_rollout, task_data, i, cfg): i for i in range(num_samples)}
            for future in as_completed(futures):
                sample_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    # Save trajectory if output_dir specified
                    if output_dir and result.steps:
                        try:
                            saved_path = save_trajectory(result, task_data, Path(output_dir), cfg)
                            print(f"✓ Saved trajectory to {saved_path}")
                        except Exception as e:
                            print(f"⚠ Failed to save trajectory: {e}")
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
        # Sort results by sample_idx for consistent ordering
        results.sort(key=lambda r: r.sample_idx)
    else:
        for i in range(num_samples):
            result = run_single_rollout(task_data, i, cfg)
            results.append(result)

            # Save trajectory if output_dir specified
            if output_dir and result.steps:
                try:
                    saved_path = save_trajectory(result, task_data, Path(output_dir), cfg)
                    print(f"✓ Saved trajectory to {saved_path}")
                except Exception as e:
                    print(f"⚠ Failed to save trajectory: {e}")

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
        tasks = list_available_tasks(cfg.data.task_jsonl_dir)
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
        task_ids = list_available_tasks(cfg.data.task_jsonl_dir)
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
