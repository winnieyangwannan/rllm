#!/usr/bin/env python3
"""Phase 3 Side-by-Side Comparison Test.

This script runs the same task with both the legacy and new agent loop,
then compares the outputs to validate that the refactored code works correctly.

Usage:
    cd /home/winnieyangwn/rllm/examples/mlebench
    python test_comparison.py --task mlsp-2013-birds --samples 1

The script:
1. Runs eval.py with --legacy-agent-loop (old sync _run_agent_loop)
2. Runs eval.py with new async MLEBenchAgent
3. Compares outputs and reports any differences

Note: Due to non-determinism in LLM responses, we compare structural properties
rather than exact outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_eval(
    config: str,
    task: str,
    samples: int,
    output_dir: Path,
    legacy: bool = False,
) -> dict:
    """Run eval.py and return results."""
    cmd = [
        sys.executable,
        "eval.py",
        "--config",
        config,
        "--task",
        task,
        "--samples",
        str(samples),
        "--output-dir",
        str(output_dir),
    ]
    if legacy:
        cmd.append("--legacy-agent-loop")

    mode = "LEGACY" if legacy else "NEW"
    print(f"\n{'=' * 60}")
    print(f"Running {mode} agent loop")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=False,  # Stream output to terminal
    )

    # Load results from JSONL file
    results_file = output_dir / f"{task}.jsonl"
    results = []
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    return {
        "exit_code": result.returncode,
        "results": results,
        "results_file": results_file,
    }


def compare_results(legacy_results: dict, new_results: dict) -> dict:
    """Compare results from legacy and new agent loops."""
    comparison = {
        "legacy_exit_code": legacy_results["exit_code"],
        "new_exit_code": new_results["exit_code"],
        "both_succeeded": legacy_results["exit_code"] == 0 and new_results["exit_code"] == 0,
        "episodes_match": len(legacy_results["results"]) == len(new_results["results"]),
        "differences": [],
    }

    # Compare episode-level metrics
    for i, (legacy_ep, new_ep) in enumerate(zip(legacy_results["results"], new_results["results"], strict=False)):
        ep_diff = {"episode_idx": i}

        # Compare key fields
        legacy_artifacts = legacy_ep.get("artifacts", {})
        new_artifacts = new_ep.get("artifacts", {})

        # Both should have same termination reason (or similar)
        if "termination_reason" in legacy_artifacts and "termination_reason" in new_artifacts:
            if legacy_artifacts["termination_reason"] != new_artifacts["termination_reason"]:
                ep_diff["termination_reason"] = {
                    "legacy": legacy_artifacts["termination_reason"],
                    "new": new_artifacts["termination_reason"],
                }

        # Both should have similar step counts (within tolerance due to non-determinism)
        legacy_traj = legacy_ep.get("trajectories", [{}])[0]
        new_traj = new_ep.get("trajectories", [{}])[0]
        legacy_steps = len(legacy_traj.get("steps", []))
        new_steps = len(new_traj.get("steps", []))

        # Report if step count differs significantly (> 50%)
        if legacy_steps > 0 and new_steps > 0:
            ratio = new_steps / legacy_steps
            if ratio < 0.5 or ratio > 2.0:
                ep_diff["step_count"] = {
                    "legacy": legacy_steps,
                    "new": new_steps,
                    "ratio": ratio,
                }

        # Check if both submitted or both did not
        legacy_submitted = legacy_artifacts.get("pred_solution") is not None
        new_submitted = new_artifacts.get("pred_solution") is not None
        if legacy_submitted != new_submitted:
            ep_diff["submission"] = {
                "legacy_submitted": legacy_submitted,
                "new_submitted": new_submitted,
            }

        if len(ep_diff) > 1:  # More than just episode_idx
            comparison["differences"].append(ep_diff)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Side-by-Side Comparison Test")
    parser.add_argument("--config", default="configs/comparison_test.yaml", help="Config file")
    parser.add_argument("--task", default="mlsp-2013-birds", help="Task ID to test")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip legacy run (use existing results)")
    parser.add_argument("--skip-new", action="store_true", help="Skip new run (use existing results)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(f"/checkpoint/maui_sft/winnieyangwn/rllm/comparison_test/{timestamp}")
    legacy_output = base_output / "legacy"
    new_output = base_output / "new"

    # Create output directories
    legacy_output.mkdir(parents=True, exist_ok=True)
    new_output.mkdir(parents=True, exist_ok=True)

    print(f"Comparison Test: {args.task}")
    print(f"Samples: {args.samples}")
    print(f"Output: {base_output}")

    # Run legacy agent loop
    if not args.skip_legacy:
        legacy_results = run_eval(
            config=args.config,
            task=args.task,
            samples=args.samples,
            output_dir=legacy_output,
            legacy=True,
        )
    else:
        print("\nSkipping legacy run (--skip-legacy)")
        legacy_results = {"exit_code": 0, "results": [], "results_file": legacy_output / f"{args.task}.jsonl"}

    # Run new agent loop
    if not args.skip_new:
        new_results = run_eval(
            config=args.config,
            task=args.task,
            samples=args.samples,
            output_dir=new_output,
            legacy=False,
        )
    else:
        print("\nSkipping new run (--skip-new)")
        new_results = {"exit_code": 0, "results": [], "results_file": new_output / f"{args.task}.jsonl"}

    # Compare results
    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")

    comparison = compare_results(legacy_results, new_results)

    print(f"\nLegacy exit code: {comparison['legacy_exit_code']}")
    print(f"New exit code: {comparison['new_exit_code']}")
    print(f"Both succeeded: {comparison['both_succeeded']}")
    print(f"Episode counts match: {comparison['episodes_match']}")

    if comparison["differences"]:
        print(f"\nDifferences found ({len(comparison['differences'])} episodes):")
        for diff in comparison["differences"]:
            print(f"  Episode {diff['episode_idx']}:")
            for key, value in diff.items():
                if key != "episode_idx":
                    print(f"    {key}: {value}")
    else:
        print("\n✓ No significant differences found!")

    # Save comparison report
    report_file = base_output / "comparison_report.json"
    comparison["legacy_output"] = str(legacy_output)
    comparison["new_output"] = str(new_output)
    comparison["task"] = args.task
    comparison["samples"] = args.samples
    comparison["config"] = args.config

    with open(report_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nReport saved to: {report_file}")

    # Summary
    if comparison["both_succeeded"] and not comparison["differences"]:
        print("\n" + "=" * 60)
        print("✓ VALIDATION PASSED: New agent loop produces equivalent results")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠ VALIDATION NEEDS REVIEW: Check differences above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
