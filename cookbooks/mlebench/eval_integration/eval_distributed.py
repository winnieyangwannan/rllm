#!/usr/bin/env python3
"""Distributed MLE-bench Evaluation Script.

Multi-node version of eval.py that distributes samples across SLURM ranks.

Usage:
    # Single node (behaves like eval.py)
    python eval_distributed.py --config configs/gpt5.yaml --task mlsp-2013-birds
    
    # Multi-node via srun (3 nodes × 8 GPUs = 24 ranks)
    srun --nodes=3 --ntasks-per-node=8 python eval_distributed.py \
        --config configs/gpt5.yaml --task mlsp-2013-birds --samples 192
    
    # Each rank runs num_rollout_threads parallel rollouts (default: 8)
    # Total: 24 ranks × 8 threads = 192 parallel AgentBox containers

Parallelization Model (matching amaia-collab):
    total_parallel = nodes × gpus_per_node × num_rollout_threads
    
    Example: 3 nodes × 8 GPUs × 8 threads = 192 parallel rollouts

Sample Distribution:
    - Total samples divided evenly across ranks
    - Each rank runs its samples using ThreadPoolExecutor  
    - Rank 0 gathers results and computes final metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# Import distributed utilities (local module)
from distributed import (
    barrier,
    cleanup_distributed,
    get_global_rank,
    get_is_rank_zero,
    get_world_size,
    init_distributed,
    print_rank0,
    shard_samples,
)
from omegaconf import OmegaConf

# Import from main eval.py (avoid duplication)
sys.path.insert(0, str(Path(__file__).parent))
from eval import (
    EvalResult,
    load_config,
    load_task_from_jsonl,
    run_single_rollout,
    save_trajectory,
)


@dataclass
class DistributedEvalResult:
    """Container for distributed evaluation results."""

    task_id: str
    total_samples: int
    world_size: int
    results: list[EvalResult]
    mean_percentile: float
    success_rate: float


def run_task_eval_distributed(
    task_id: str,
    cfg: OmegaConf,
    num_samples: int | None = None,
    output_dir: str | None = None,
    num_rollout_threads: int = 8,
) -> DistributedEvalResult:
    """Run distributed evaluation on a single task.

    Each rank processes its shard of samples, then rank 0 gathers results.
    """
    rank = get_global_rank()
    world_size = get_world_size()

    # Use CLI overrides or config defaults
    num_samples = num_samples or cfg.eval.samples_per_prompt
    output_dir = output_dir or cfg.eval.get("output_dir")
    max_workers = cfg.eval.get("max_workers") or num_rollout_threads

    # Print config (rank 0 only)
    print_rank0("\n" + "=" * 70)
    print_rank0(f"DISTRIBUTED TASK EVAL: {task_id}")
    print_rank0("=" * 70)
    print_rank0("Config:")
    print_rank0(f"  model: {cfg.model.name}")
    print_rank0(f"  total_samples: {num_samples}")
    print_rank0(f"  world_size: {world_size}")
    print_rank0(f"  num_rollout_threads per rank: {max_workers}")
    print_rank0(f"  total_parallel: {world_size * max_workers}")
    print_rank0(f"  manager_uri: {cfg.sandbox.manager_uri}")
    print_rank0(f"  output_dir: {output_dir or 'None'}")

    # Calculate this rank's sample shard
    start_idx, end_idx = shard_samples(num_samples, rank, world_size)
    local_samples = end_idx - start_idx
    print(f"[Rank {rank}/{world_size}] Processing samples {start_idx}-{end_idx - 1} ({local_samples} samples)")

    # Load task data (all ranks)
    task_data = load_task_from_jsonl(task_id, cfg.data.task_jsonl_dir)
    print(f"[Rank {rank}] Loaded task: {task_id}")

    # Synchronize before starting rollouts
    barrier()

    # Run rollouts for this rank's shard
    results = []
    print(f"[Rank {rank}] Running {local_samples} rollouts with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs for this rank's sample range
        futures = {}
        for global_idx in range(start_idx, end_idx):
            future = executor.submit(run_single_rollout, task_data, global_idx, cfg)
            futures[future] = global_idx

        # Collect results
        for future in as_completed(futures):
            global_idx = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Save trajectory if output_dir specified
                if output_dir and result.steps:
                    try:
                        # Include rank in filename to avoid collisions
                        save_trajectory(result, task_data, Path(output_dir), cfg)
                        print(f"[Rank {rank}] ✓ Saved trajectory for sample {global_idx}")
                    except Exception as e:
                        print(f"[Rank {rank}] ⚠ Failed to save trajectory: {e}")

                # Progress update
                completed = len(results)
                print(f"[Rank {rank}] Sample {global_idx}: {'✓' if result.success else '✗'} percentile={result.percentile:.4f} ({completed}/{local_samples} done)")

            except Exception as e:
                print(f"[Rank {rank}] ⚠ Sample {global_idx} failed: {e}")
                results.append(
                    EvalResult(
                        task_id=task_id,
                        sample_idx=global_idx,
                        success=False,
                        percentile=0.0,
                        score=0.0,
                        error=str(e),
                    )
                )

    # Synchronize before gathering
    barrier()
    print(f"[Rank {rank}] Completed {len(results)} rollouts, gathering results...")

    # Gather results to rank 0
    # For simplicity, use file-based gathering (works without complex distributed setup)
    if output_dir:
        # Save local results to a rank-specific file
        local_results_path = Path(output_dir) / f"results_rank{rank}.json"
        local_data = [asdict(r) for r in results]
        with open(local_results_path, "w") as f:
            json.dump(local_data, f)
        print(f"[Rank {rank}] Saved local results to {local_results_path}")

    barrier()

    # Rank 0 aggregates all results
    if get_is_rank_zero() and output_dir:
        all_results = []
        for r in range(world_size):
            results_path = Path(output_dir) / f"results_rank{r}.json"
            if results_path.exists():
                with open(results_path) as f:
                    rank_results = json.load(f)
                    all_results.extend([EvalResult(**r) for r in rank_results])
                # Clean up rank file
                results_path.unlink()

        # Calculate aggregate metrics
        successful = [r for r in all_results if r.success]
        mean_percentile = sum(r.percentile for r in all_results) / len(all_results) if all_results else 0
        success_rate = len(successful) / len(all_results) if all_results else 0

        print("\n" + "=" * 70)
        print("DISTRIBUTED RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total samples: {len(all_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Mean percentile: {mean_percentile:.4f}")
        print("=" * 70)

        # Save aggregate results
        summary = {
            "task_id": task_id,
            "total_samples": len(all_results),
            "world_size": world_size,
            "successful": len(successful),
            "success_rate": success_rate,
            "mean_percentile": mean_percentile,
            "results": [asdict(r) for r in all_results],
        }
        summary_path = Path(output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_path}")

        return DistributedEvalResult(
            task_id=task_id,
            total_samples=len(all_results),
            world_size=world_size,
            results=all_results,
            mean_percentile=mean_percentile,
            success_rate=success_rate,
        )
    else:
        # Non-rank-0 processes return partial results
        mean_percentile = sum(r.percentile for r in results) / len(results) if results else 0
        success_rate = len([r for r in results if r.success]) / len(results) if results else 0
        return DistributedEvalResult(
            task_id=task_id,
            total_samples=len(results),
            world_size=world_size,
            results=results,
            mean_percentile=mean_percentile,
            success_rate=success_rate,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Distributed MLE-bench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single node (24 threads)
    python eval_distributed.py --config configs/gpt5.yaml --task mlsp-2013-birds --threads 24

    # Multi-node via srun (3 nodes × 8 ranks = 24 processes, each with 8 threads = 192 parallel)
    srun --nodes=3 --ntasks-per-node=8 python eval_distributed.py \\
        --config configs/gpt5.yaml --task mlsp-2013-birds --samples 192 --threads 8
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task", type=str, help="Single task ID to evaluate")
    parser.add_argument("--samples", type=int, help="Total samples (distributed across ranks)")
    parser.add_argument("--threads", type=int, default=8, help="Rollout threads per rank (default: 8)")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Initialize distributed if running under SLURM
    is_distributed = init_distributed(backend="gloo")

    rank = get_global_rank()
    world_size = get_world_size()

    print_rank0(f"\n{'=' * 70}")
    print_rank0("DISTRIBUTED MLE-BENCH EVALUATION")
    print_rank0(f"{'=' * 70}")
    print_rank0(f"Distributed: {is_distributed}")
    print_rank0(f"World size: {world_size}")
    print_rank0(f"Threads per rank: {args.threads}")
    print_rank0(f"Total parallel capacity: {world_size * args.threads}")

    # Load config
    print(f"[Rank {rank}] Loading config from {args.config}...")
    cfg = load_config(args.config)

    # Run evaluation
    try:
        result = run_task_eval_distributed(
            task_id=args.task,
            cfg=cfg,
            num_samples=args.samples,
            output_dir=args.output_dir,
            num_rollout_threads=args.threads,
        )

        if get_is_rank_zero():
            print("\n✓ Evaluation complete!")
            print(f"  Mean percentile: {result.mean_percentile:.4f}")
            print(f"  Success rate: {result.success_rate:.2%}")

    finally:
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    sys.exit(main())
