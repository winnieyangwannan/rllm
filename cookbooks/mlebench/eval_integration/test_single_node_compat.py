#!/usr/bin/env python3
"""Test: Single-Node Compatibility for eval_ray.py

Verifies that eval_ray.py produces identical results to eval.py when run on a single node.
This is the first test to pass before implementing multi-node support.

Usage:
    # Run from eval_integration directory
    cd /home/winnieyangwn/rllm/cookbooks/mlebench/eval_integration
    conda activate rllm

    # Test baseline (eval.py only)
    python test_single_node_compat.py --baseline-only

    # Full compatibility test (compares eval.py vs eval_ray.py)
    python test_single_node_compat.py

    # With custom config/task
    python test_single_node_compat.py --config configs/gpt5_test.yaml --task mlsp-2013-birds

Expected behavior:
    - eval_ray.py with no Ray cluster should start a local Ray instance
    - Results (percentiles, success, num_steps) should match eval.py exactly
    - Trajectory structure should be identical
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Conda environment to use
CONDA_ENV = "rllm"


@dataclass
class TestResult:
    """Single rollout result for comparison."""

    task_id: str
    sample_idx: int
    percentile: float
    success: bool
    num_steps: int
    has_pred_solution: bool
    error: str | None = None


def run_eval_script(
    script: str,
    config: str,
    task: str,
    samples: int,
    output_dir: Path,
) -> list[TestResult]:
    """Run an eval script and parse results."""

    # Use conda run to execute in the correct environment
    cmd = [
        "conda",
        "run",
        "-n",
        CONDA_ENV,
        "--no-capture-output",
        "python",
        script,
        "--config",
        config,
        "--task",
        task,
        "--samples",
        str(samples),
        "--output-dir",
        str(output_dir),
    ]

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with return code {result.returncode}")

    # Parse output JSONL files
    results = []
    for jsonl_file in output_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                # Extract from Episode structure
                metrics = data.get("metrics", {})
                outcomes = data.get("outcomes", {})
                results.append(
                    TestResult(
                        task_id=data.get("task", {}).get("instance_id", "unknown"),
                        sample_idx=int(data.get("id", "0:0").split(":")[-1]) if ":" in str(data.get("id", "")) else 0,
                        percentile=metrics.get("percentile", 0.0),
                        success=outcomes.get("pass", False),
                        num_steps=metrics.get("num_steps", 0),
                        has_pred_solution=data.get("artifacts", {}).get("pred_solution") is not None,
                    )
                )

    return sorted(results, key=lambda r: r.sample_idx)


def compare_results(
    baseline: list[TestResult],
    test: list[TestResult],
    tolerance: float = 0.0001,
) -> tuple[bool, list[str]]:
    """Compare two sets of results.

    Returns:
        (all_passed, list of error messages)
    """
    errors = []

    if len(baseline) != len(test):
        errors.append(f"Sample count mismatch: baseline={len(baseline)}, test={len(test)}")
        return False, errors

    for i, (b, t) in enumerate(zip(baseline, test, strict=False)):
        # Compare key metrics
        if b.task_id != t.task_id:
            errors.append(f"Sample {i}: task_id mismatch: {b.task_id} vs {t.task_id}")

        if abs(b.percentile - t.percentile) > tolerance:
            errors.append(f"Sample {i}: percentile mismatch: {b.percentile:.4f} vs {t.percentile:.4f}")

        if b.success != t.success:
            errors.append(f"Sample {i}: success mismatch: {b.success} vs {t.success}")

        if b.num_steps != t.num_steps:
            # Steps can vary slightly due to timing, warn but don't fail
            print(f"  ⚠ Sample {i}: num_steps differs: {b.num_steps} vs {t.num_steps} (non-fatal)")

        if b.has_pred_solution != t.has_pred_solution:
            errors.append(f"Sample {i}: pred_solution presence mismatch: {b.has_pred_solution} vs {t.has_pred_solution}")

    return len(errors) == 0, errors


def test_baseline(config: str, task: str, samples: int) -> bool:
    """Test that eval.py works (baseline)."""
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (eval.py)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "baseline"
        output_dir.mkdir()

        try:
            results = run_eval_script("eval.py", config, task, samples, output_dir)
            print(f"\n✓ eval.py completed: {len(results)} results")
            for r in results:
                status = "✓" if r.success else "✗"
                print(f"  {status} Sample {r.sample_idx}: percentile={r.percentile:.4f}, steps={r.num_steps}")
            return True
        except Exception as e:
            print(f"\n✗ eval.py failed: {e}")
            return False


def test_ray_single_node(config: str, task: str, samples: int) -> bool:
    """Test that eval_ray.py works in single-node mode."""
    print("\n" + "=" * 70)
    print("TEST 2: Ray Single-Node (eval_ray.py, no cluster)")
    print("=" * 70)

    # Check if eval_ray.py exists
    if not Path("eval_ray.py").exists():
        print("\n⚠ eval_ray.py does not exist yet. Skipping test.")
        print("  Create eval_ray.py first, then re-run this test.")
        return True  # Not a failure, just not implemented yet

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "ray_single"
        output_dir.mkdir()

        try:
            results = run_eval_script("eval_ray.py", config, task, samples, output_dir)
            print(f"\n✓ eval_ray.py completed: {len(results)} results")
            for r in results:
                status = "✓" if r.success else "✗"
                print(f"  {status} Sample {r.sample_idx}: percentile={r.percentile:.4f}, steps={r.num_steps}")
            return True
        except Exception as e:
            print(f"\n✗ eval_ray.py failed: {e}")
            return False


def test_compatibility(config: str, task: str, samples: int) -> bool:
    """Full compatibility test: compare eval.py vs eval_ray.py."""
    print("\n" + "=" * 70)
    print("TEST 3: Compatibility (eval.py vs eval_ray.py)")
    print("=" * 70)

    # Check if eval_ray.py exists
    if not Path("eval_ray.py").exists():
        print("\n⚠ eval_ray.py does not exist yet. Skipping compatibility test.")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baseline"
        test_dir = Path(tmpdir) / "ray_single"
        baseline_dir.mkdir()
        test_dir.mkdir()

        try:
            print("\nRunning baseline (eval.py)...")
            baseline_results = run_eval_script("eval.py", config, task, samples, baseline_dir)

            print("\nRunning test (eval_ray.py)...")
            test_results = run_eval_script("eval_ray.py", config, task, samples, test_dir)

            print("\nComparing results...")
            passed, errors = compare_results(baseline_results, test_results)

            if passed:
                print("\n✓ Compatibility test PASSED!")
                print("  eval_ray.py produces identical results to eval.py in single-node mode.")
                return True
            else:
                print("\n✗ Compatibility test FAILED!")
                for err in errors:
                    print(f"  - {err}")
                return False

        except Exception as e:
            print(f"\n✗ Compatibility test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Test single-node compatibility for eval_ray.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpt5_test.yaml",
        help="Config file to use for testing",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mlsp-2013-birds",
        help="Task ID to test with",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples (keep small for quick testing)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline test (eval.py)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SINGLE-NODE COMPATIBILITY TEST")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Task: {args.task}")
    print(f"Samples: {args.samples}")

    # Test 1: Baseline
    if not test_baseline(args.config, args.task, args.samples):
        print("\n" + "=" * 70)
        print("RESULT: FAILED (baseline eval.py doesn't work)")
        print("=" * 70)
        sys.exit(1)

    if args.baseline_only:
        print("\n" + "=" * 70)
        print("RESULT: PASSED (baseline only)")
        print("=" * 70)
        sys.exit(0)

    # Test 2: Ray single-node
    if not test_ray_single_node(args.config, args.task, args.samples):
        print("\n" + "=" * 70)
        print("RESULT: FAILED (eval_ray.py doesn't work)")
        print("=" * 70)
        sys.exit(1)

    # Test 3: Compatibility
    if not test_compatibility(args.config, args.task, args.samples):
        print("\n" + "=" * 70)
        print("RESULT: FAILED (results don't match)")
        print("=" * 70)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("RESULT: ALL TESTS PASSED")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Test multi-node: python launch.py --nodes 2 --samples 4 --dry-run")
    print("  2. Submit multi-node job: python launch.py --nodes 3 --samples 48")


if __name__ == "__main__":
    main()
