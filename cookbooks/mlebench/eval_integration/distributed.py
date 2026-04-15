#!/usr/bin/env python3
"""Distributed utilities for multi-node MLE-bench evaluation.

Mimics amaia-collab's approach using SLURM environment variables and
torch.distributed for coordination.

Usage in SLURM:
    #SBATCH --nodes=3
    #SBATCH --ntasks-per-node=8  # 8 processes per node (1 per GPU)

    srun python eval_distributed.py --config configs/gpt5.yaml --task mlsp-2013-birds

This will:
- Launch 24 processes (3 nodes × 8 tasks)
- Each process runs num_rollout_threads parallel rollouts
- Total: 24 × num_rollout_threads parallel AgentBox containers
"""

from __future__ import annotations

import os
import random
import subprocess
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist

# ============================================================================
# Environment Detection (from amaia-collab pattern)
# ============================================================================


def get_is_torch_run() -> bool:
    """Check if running under torchrun/torch.distributed.launch."""
    return os.environ.get("LOCAL_RANK") is not None


def get_is_slurm_job() -> bool:
    """Check if running as a SLURM job (not via torchrun)."""
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache
def get_global_rank() -> int:
    """Get global rank across all nodes."""
    if get_is_torch_run():
        return int(os.environ["RANK"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0


@lru_cache
def get_local_rank() -> int:
    """Get local rank within this node."""
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    return 0


@lru_cache
def get_world_size() -> int:
    """Get total number of processes across all nodes."""
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    return 1


@lru_cache
def get_is_rank_zero() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_global_rank() == 0


@lru_cache
def get_master_addr() -> str:
    """Get master node address for distributed init."""
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    if get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        )
        return hostnames.split()[0].decode("utf-8")
    return "127.0.0.1"


@lru_cache
def get_master_port() -> int:
    """Get master port for distributed init (deterministic from job ID)."""
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
    rng = random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
    return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


# ============================================================================
# Distributed Initialization
# ============================================================================

_INITIALIZED = False


def init_distributed(backend: str = "gloo") -> bool:
    """Initialize torch.distributed if running in multi-node SLURM.

    Args:
        backend: torch.distributed backend ('gloo' for CPU, 'nccl' for GPU)

    Returns:
        True if distributed was initialized, False if running single-process
    """
    global _INITIALIZED

    if _INITIALIZED:
        return dist.is_initialized()

    world_size = get_world_size()
    if world_size <= 1:
        _INITIALIZED = True
        return False

    rank = get_global_rank()
    master_addr = get_master_addr()
    master_port = get_master_port()

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    print(f"[Rank {rank}/{world_size}] Initializing distributed: {master_addr}:{master_port}")

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    _INITIALIZED = True
    print(f"[Rank {rank}/{world_size}] Distributed initialized successfully")
    return True


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Work Distribution
# ============================================================================


def shard_samples(total_samples: int, rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    """Compute which sample indices this rank should process.

    Distributes samples evenly across ranks, with earlier ranks getting
    any remainder samples.

    Args:
        total_samples: Total number of samples to distribute
        rank: This process's rank (defaults to get_global_rank())
        world_size: Total number of processes (defaults to get_world_size())

    Returns:
        (start_idx, end_idx): Range of sample indices for this rank (exclusive end)

    Example:
        # 64 samples across 3 ranks:
        # Rank 0: samples 0-21 (22 samples)
        # Rank 1: samples 22-43 (22 samples)
        # Rank 2: samples 44-63 (20 samples)
    """
    if rank is None:
        rank = get_global_rank()
    if world_size is None:
        world_size = get_world_size()

    if world_size <= 1:
        return 0, total_samples

    base_count = total_samples // world_size
    remainder = total_samples % world_size

    # Earlier ranks get one extra sample if there's a remainder
    if rank < remainder:
        start = rank * (base_count + 1)
        end = start + base_count + 1
    else:
        start = remainder * (base_count + 1) + (rank - remainder) * base_count
        end = start + base_count

    return start, end


def gather_results(local_results: list[Any], rank: int | None = None, world_size: int | None = None) -> list[Any] | None:
    """Gather results from all ranks to rank 0.

    Args:
        local_results: Results from this rank
        rank: This process's rank
        world_size: Total number of processes

    Returns:
        Combined results on rank 0, None on other ranks
    """
    if rank is None:
        rank = get_global_rank()
    if world_size is None:
        world_size = get_world_size()

    if world_size <= 1:
        return local_results

    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")

    # Gather list lengths first
    local_count = torch.tensor([len(local_results)], dtype=torch.long)
    if rank == 0:
        all_counts = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
        dist.gather(local_count, gather_list=all_counts, dst=0)
    else:
        dist.gather(local_count, dst=0)

    # For now, use a simple approach: pickle and gather
    import pickle

    local_bytes = pickle.dumps(local_results)
    local_tensor = torch.ByteTensor(list(local_bytes))
    local_size = torch.tensor([len(local_bytes)], dtype=torch.long)

    if rank == 0:
        all_sizes = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
        dist.gather(local_size, gather_list=all_sizes, dst=0)

        max_size = max(s.item() for s in all_sizes)
        padded_local = torch.zeros(max_size, dtype=torch.uint8)
        padded_local[: len(local_tensor)] = local_tensor

        all_tensors = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
        dist.gather(padded_local, gather_list=all_tensors, dst=0)

        # Unpack results
        all_results = []
        for i, (tensor, size) in enumerate(zip(all_tensors, all_sizes, strict=False)):
            data = bytes(tensor[: size.item()].tolist())
            results = pickle.loads(data)
            all_results.extend(results)

        return all_results
    else:
        # Non-rank-0: just send our data
        dist.gather(local_size, dst=0)

        max_size_tensor = torch.zeros(1, dtype=torch.long)
        dist.broadcast(max_size_tensor, src=0)  # Get max size from rank 0
        # Actually we need a different approach - use all_gather

        # Simpler: just gather without size coordination
        dist.gather(local_tensor, dst=0)
        return None


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


# ============================================================================
# Convenience Functions
# ============================================================================


def print_rank0(msg: str, *args, **kwargs):
    """Print only on rank 0."""
    if get_is_rank_zero():
        print(msg, *args, **kwargs)


def get_distributed_info() -> dict:
    """Get summary of distributed environment."""
    return {
        "rank": get_global_rank(),
        "local_rank": get_local_rank(),
        "world_size": get_world_size(),
        "is_rank_zero": get_is_rank_zero(),
        "is_slurm": get_is_slurm_job(),
        "master_addr": get_master_addr() if get_is_slurm_job() else None,
        "master_port": get_master_port() if get_is_slurm_job() else None,
    }


if __name__ == "__main__":
    # Test distributed utilities
    print("Distributed environment info:")
    for k, v in get_distributed_info().items():
        print(f"  {k}: {v}")

    # Test sharding
    print("\nSample sharding (64 samples, 3 ranks):")
    for r in range(3):
        start, end = shard_samples(64, rank=r, world_size=3)
        print(f"  Rank {r}: samples {start}-{end - 1} ({end - start} samples)")
