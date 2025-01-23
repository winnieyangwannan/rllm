"""Distributed inference sampler using Ray.

This module provides distributed inference capabilities by managing multiple worker processes
across GPUs using Ray. It supports different inference backends (SGLang, vLLM) and handles
worker lifecycle management and request routing.

Key Components:
- DistributedSampler: Main class managing worker pool and request distribution
- EngineBackend: Enum defining supported inference backends

Features:
- Dynamic worker allocation and management
- Load balancing across workers
- Fault tolerance and worker recovery
- Resource cleanup on shutdown
"""
from enum import Enum
import os
import signal
import threading
from typing import Dict, List, Any

import ray
import torch

from rllm.globals import BASE_SAMPLER_PORT
from rllm.sampler.engine import RaySGLangWorker, RayVLLMWorker
from rllm.sampler.sampler_types import SampleBatch


class EngineBackend(Enum):
    """Supported inference engine backends."""
    SGLANG = "SGLANG"
    VLLM = "VLLM"


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Workers will continue running in background.")
    print("To shutdown workers, call shutdown().")
    # Unregister the signal handler to prevent further interrupts
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Exit without calling sys.exit() to keep Ray actors running
    os._exit(0)

class DistributedSampler:
    """Manages distributed inference across multiple workers using Ray.

    This class handles:
    - Starting/stopping Ray cluster
    - Managing worker processes
    - Load balancing inference requests
    - Resource cleanup

    Attributes:
        model: Name or path of model to load
        backend: Inference engine backend (SGLang/vLLM)
        num_workers: Number of worker processes
        workers: List of Ray actor handles
        active_requests: Count of active requests per worker
        worker_ports: Port numbers for each worker
    """

    WORKER_NAME_TEMPLATE = "persistent_worker_{}"
    NAMESPACE = "workers"

    def __init__(self, num_workers: int = 1, backend: str = "sglang", **model_kwargs) -> None:
        """Initialize the distributed inference system.

        Args:
            num_workers: Number of worker processes to launch
            backend: Inference engine backend ("sglang" or "vllm")
            **model_kwargs: Model configuration parameters including:
                - model/model_path: Model name or path
                - tensor_parallel_size: Number of GPUs per worker
                - Other engine-specific parameters
        """
        # Move signal handler setup before Ray initialization
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        self.backend = EngineBackend(backend.upper())
        self.num_workers = num_workers
        self.model_kwargs = model_kwargs
        self.active_requests = {}
        self.workers = []
        self.worker_ports = []
        self._worker_lock = threading.Lock()

        # Initialize Ray cluster
        try:
            ray.init(address="auto", namespace=self.NAMESPACE)
            print("Connected to existing Ray cluster")
        except ConnectionError:
            print("Starting new Ray cluster...")
            total_gpus = torch.cuda.device_count()
            total_cpus = os.cpu_count()
            os.system(f"ray start --head --num-cpus={total_cpus} --num-gpus={total_gpus}")
            ray.init(address="auto", namespace=self.NAMESPACE)

        # Initialize workers
        self.workers = self.get_or_launch_workers()
        print("All workers initialized successfully")

    def get_or_launch_workers(self) -> List[Any]:
        """Launch new workers or connect to existing ones.

        Returns:
            List of Ray actor handles for workers
        """
        worker_refs = []
        new_worker_refs = []
        self.worker_ports = [BASE_SAMPLER_PORT + i for i in range(self.num_workers)]

        # Select worker class based on backend
        if self.backend == EngineBackend.SGLANG:
            WorkerClass = RaySGLangWorker
        elif self.backend == EngineBackend.VLLM:
            WorkerClass = RayVLLMWorker
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        for i in range(self.num_workers):
            worker_name = self.WORKER_NAME_TEMPLATE.format(i)
            try:
                worker = ray.get_actor(worker_name, namespace=self.NAMESPACE)
                print(f"Found existing worker: {worker_name}")
            except ValueError:
                print(f"Creating new worker: {worker_name}")
                # Create worker with fault tolerance settings and signal handling
                worker = WorkerClass.options(
                    name=worker_name,
                    lifetime="detached",
                    namespace=self.NAMESPACE,
                    num_gpus=self.model_kwargs.get("tensor_parallel_size", 1),
                    num_cpus=8,
                    max_concurrency=1024,
                    runtime_env={
                        "env_vars": {
                            "PYTHONUNBUFFERED": "1",
                            "IGNORE_SIGINT": "1"  # Environment variable to ignore SIGINT in worker
                        }
                    }
                ).remote(
                    port=self.worker_ports[i],
                    **self.model_kwargs
                )
                new_worker_refs.append(worker)
            worker_refs.append(worker)
            self.active_requests[i] = 0

        # Start new workers if any
        if new_worker_refs:
            [w.start_server.remote() for w in new_worker_refs]
        ray.get([w.wait_for_server.remote() for w in new_worker_refs])
        return worker_refs

    def _get_least_busy_worker(self) -> int:
        """Get index of worker with fewest active requests.

        Returns:
            Index of least busy worker
        """
        with self._worker_lock:
            least_busy_idx = min(self.active_requests, key=self.active_requests.get)
            self.active_requests[least_busy_idx] += 1
            return least_busy_idx

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> SampleBatch:
        """Send chat completion request to least busy worker.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for chat completion API

        Returns:
            SampleBatch containing response and metrics

        Raises:
            Exception: If worker request fails
        """
        worker_idx = self._get_least_busy_worker()
        print(f"Using worker {worker_idx}; {self.active_requests}")

        try:
            sample_batch = ray.get(self.workers[worker_idx].chat_completion.remote(
                messages=messages, **kwargs))
            with self._worker_lock:
                self.active_requests[worker_idx] -= 1
            return sample_batch
        except Exception as e:
            with self._worker_lock:
                self.active_requests[worker_idx] -= 1
            raise e

    def restart(self) -> None:
        """Restart all workers."""
        self.shutdown()
        self.get_or_launch_workers()

    def shutdown(self) -> None:
        """Shutdown distributed system and cleanup resources.

        Terminates all worker processes and cleans up Ray actors.
        """
        print("Shutting down workers...")
        for worker in self.workers:
            ray.get(worker.shutdown.remote())
            ray.kill(worker)


if __name__ == "__main__":
    sampler = DistributedSampler(
        num_workers=1,
        tensor_parallel_size=2,
        backend="sglang",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    print(sampler.chat_completion([{
        "role": "user",
        "content": "Find the minimum value of $\\frac{9x^2\\sin^2 x + 4}{x\\sin x}$ for $0 < x < \\pi$."
    }]))
    sampler.shutdown()
