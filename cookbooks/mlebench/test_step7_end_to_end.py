#!/usr/bin/env python3
"""
deactivate  # deactivate the .venv
conda activate rllm  # ensure conda rllm is active

Step 7: End-to-End Eval Run (Agent + Evaluator)

Tests the full pipeline:
1. Load task from JSONL
2. Start AgentBox container
3. Run agent loop (up to max_turns)
4. Agent calls bash/edit/create/submit tools
5. When terminal → pass to evaluator
6. Evaluator returns percentile

Usage:
    # Basic usage (parallel by default, 2 samples)
    python test_step7_end_to_end.py --task mlsp-2013-birds

    # Run sequentially instead of parallel
    python test_step7_end_to_end.py --task mlsp-2013-birds --sequential

    # Run 4 parallel samples
    python test_step7_end_to_end.py --task mlsp-2013-birds --samples 4

    # Use code submission mode instead of CSV
    python test_step7_end_to_end.py --task spooky-author-identification --submit-file code

    # Custom manager URI
    python test_step7_end_to_end.py --task mlsp-2013-birds --manager-uri h200-137-000-067:42499

    # Custom output directory for trajectories
    python test_step7_end_to_end.py --task mlsp-2013-birds --output-dir /path/to/output

    # Skip saving trajectories
    python test_step7_end_to_end.py --task mlsp-2013-birds --no-save

Config (for testing):
    samples_per_prompt = 2
    rollout_timeout = 20 min (1200s)
    session_timeout = 5 min (300s)
    max_turns = 128
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add mle_agent to path
sys.path.insert(0, "/home/winnieyangwn/rllm/agenthub/mle_agent")

# Import rLLM types for trajectory saving
from rllm.types import Episode, Trajectory

# ============================================================================
# CONFIG
# ============================================================================

MANAGER_URI = "h200-137-000-067:42499"

# Azure OpenAI
AZURE_ENDPOINT = "https://azure-services-fair-openai1-eastus2n3.azure-api.net"
LLM_MODEL = "gpt-5"
LLM_API_KEY = "73afb4e502de426c8ea645416de6ec0b"
LLM_API_VERSION = "2025-03-01-preview"

# MLE-bench data
MLE_BENCH_DATA_DIR = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"
TASK_JSONL_DIR = "/checkpoint/maui_sft/winnieyangwn/datasets"

# Superimage config (matches amaia-collab)
SUPERIMAGE_DIR = "/checkpoint/maui_sft/shared/sif"
SUPERIMAGE_VERSION = "2025-05-02v2"
SUPERIMAGE_OVERLAY = "/checkpoint/fair-maui-hs/hotfix/kniu.2025-09-19.cache.overlay.ext3.img"

# Test config (reduced for testing)
SAMPLES_PER_PROMPT = 2
ROLLOUT_TIMEOUT = 43200  # 12 hours
SESSION_TIMEOUT = 1200  # 20 minutes per bash call
MAX_TURNS = 128
CONTEXT_SIZE = 98304


# Default output directory for trajectories
DEFAULT_OUTPUT_DIR = "/checkpoint/maui_sft/winnieyangwn/RLLM"


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


def load_task_from_jsonl(task_id: str) -> dict[str, Any]:
    """Load task instance from JSONL file."""
    jsonl_path = Path(TASK_JSONL_DIR) / f"{task_id}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("instance_id") == task_id:
                return data

    raise ValueError(f"Task {task_id} not found in {jsonl_path}")


def run_single_rollout(
    task_data: dict[str, Any],
    sample_idx: int,
    manager_uri: str,
    submit_file: str = "csv",
) -> EvalResult:
    """Run a single agent rollout and evaluate.

    Args:
        task_data: Task dictionary with instance_id, task_description, etc.
        sample_idx: Index of this sample (0-based).
        manager_uri: AgentBox manager URI.
        submit_file: Submission mode - 'code' or 'csv'.
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
    data_path = f"{MLE_BENCH_DATA_DIR}/{task_id}/prepared/public"
    print(f"Data path: {data_path}")

    # Build container config with data mount (matches amaia-collab)
    container_config = ContainerConfig(
        superimage_directory=SUPERIMAGE_DIR,
        superimage_version=SUPERIMAGE_VERSION,
        container_runtime="apptainer",
        read_only_overlays=[SUPERIMAGE_OVERLAY],
        read_only_binds={data_path: "/root/data"},  # Mount task data
        working_dir="/workspace",
        env={"HF_HUB_OFFLINE": "1", "NLTK_DATA": "/root/.nltk_data"},
    )

    # Create sandbox with container config
    sandbox = AgentBoxSandbox(
        name=f"e2e-{task_id}-{sample_idx}",
        manager_uri=manager_uri,
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
            timeout_min=int(SESSION_TIMEOUT / 60),
            context_size=CONTEXT_SIZE,
            eval_timeout_hrs=int(ROLLOUT_TIMEOUT / 3600),
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
            azure_endpoint=AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )
        print("✓ Created AzureOpenAI client")

        # Run agent loop
        print(f"Starting agent loop (max_turns={MAX_TURNS}, timeout={ROLLOUT_TIMEOUT}s)...")
        print(f"Tools: bash, edit, create, submit ({submit_file} mode), check_submission_validity")
        steps, final_messages, pred_solution = _run_agent_loop(
            client=client,
            model=LLM_MODEL,
            messages=messages,
            sandbox=sandbox,
            max_turns=MAX_TURNS,
            session_timeout=SESSION_TIMEOUT,
            rollout_timeout=ROLLOUT_TIMEOUT,
            temperature=1.0,
            check_submission_validity=True,
            task_id=task_id,
            mle_bench_data_dir=MLE_BENCH_DATA_DIR,
            submit_file=submit_file,
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
            mle_bench_data_dir=MLE_BENCH_DATA_DIR,
        )

        # Create mock Episode-like object for evaluator
        # Evaluator expects: task dict + episode with artifacts
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


def save_trajectory(result: EvalResult, task_data: dict, output_dir: Path) -> str:
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


def run_end_to_end_eval(
    task_id: str,
    num_samples: int = SAMPLES_PER_PROMPT,
    manager_uri: str = MANAGER_URI,
    output_dir: str | None = None,
    submit_file: str = "csv",
    parallel: bool = False,
):
    """Run end-to-end evaluation on a task."""
    print("\n" + "=" * 70)
    print(f"STEP 7: END-TO-END EVAL - {task_id}")
    print("=" * 70)
    print("Config:")
    print(f"  samples_per_prompt: {num_samples}")
    print(f"  rollout_timeout: {ROLLOUT_TIMEOUT}s ({ROLLOUT_TIMEOUT / 60:.0f} min)")
    print(f"  session_timeout: {SESSION_TIMEOUT}s ({SESSION_TIMEOUT / 60:.0f} min)")
    print(f"  max_turns: {MAX_TURNS}")
    print(f"  manager_uri: {manager_uri}")
    print(f"  output_dir: {output_dir or 'None (no saving)'}")
    print(f"  submit_file: {submit_file}")
    print(f"  parallel: {parallel}")

    # Load task
    print("\nLoading task from JSONL...")
    task_data = load_task_from_jsonl(task_id)
    print(f"✓ Loaded task: {task_id}")
    print(f"  Difficulty: {task_data.get('difficulty', 'unknown')}")
    print(f"  Description length: {len(task_data.get('task_description', ''))} chars")

    # Run rollouts
    results = []

    if parallel and num_samples > 1:
        print(f"\nRunning {num_samples} rollouts in parallel...")
        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            futures = {executor.submit(run_single_rollout, task_data, i, manager_uri, submit_file): i for i in range(num_samples)}
            for future in as_completed(futures):
                sample_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    # Save trajectory if output_dir specified
                    if output_dir and result.steps:
                        try:
                            saved_path = save_trajectory(result, task_data, Path(output_dir))
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
            result = run_single_rollout(task_data, i, manager_uri, submit_file)
            results.append(result)

            # Save trajectory if output_dir specified
            if output_dir and result.steps:
                try:
                    saved_path = save_trajectory(result, task_data, Path(output_dir))
                    print(f"✓ Saved trajectory to {saved_path}")
                except Exception as e:
                    print(f"⚠ Failed to save trajectory: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.success]
    percentiles = [r.percentile for r in results if r.percentile > 0]

    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  Sample {r.sample_idx + 1}: {status} percentile={r.percentile:.4f}, steps={r.num_steps}, duration={r.duration:.1f}s")
        if r.error:
            print(f"    Error: {r.error[:100]}")

    print(f"\nResults: {len(successful)}/{len(results)} successful")
    if percentiles:
        print(f"Mean percentile: {sum(percentiles) / len(percentiles):.4f}")
        print(f"Max percentile: {max(percentiles):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 7: End-to-End MLE-bench Eval")
    parser.add_argument("--task", type=str, default="mlsp-2013-birds", help="Task ID (must have matching JSONL in TASK_JSONL_DIR)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_PROMPT, help=f"Number of samples to run (default: {SAMPLES_PER_PROMPT})")
    parser.add_argument("--manager-uri", type=str, default=MANAGER_URI, help="AgentBox manager URI")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save trajectories (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--no-save", action="store_true", help="Skip saving trajectories")
    parser.add_argument("--submit-file", type=str, choices=["code", "csv"], default="csv", help="Submission mode: 'code' (submit train script path) or 'csv' (submit CSV directly, default)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run rollouts in parallel (each gets its own AgentBox container). Default: True")
    parser.add_argument("--sequential", action="store_true", help="Run rollouts sequentially (overrides --parallel)")

    args = parser.parse_args()

    # Determine output directory (save by default unless --no-save)
    output_dir = None if args.no_save else args.output_dir

    # Sequential flag overrides parallel default
    parallel = args.parallel and not args.sequential

    results = run_end_to_end_eval(args.task, args.samples, args.manager_uri, output_dir, args.submit_file, parallel)

    # Exit with success if any rollout succeeded
    success = any(r.success for r in results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
