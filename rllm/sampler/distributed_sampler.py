"""
Distributed inference using vLLM and Ray.

This module provides distributed inference capabilities by running multiple vLLM servers
across Ray workers. Each worker runs an independent vLLM server that can handle requests
in parallel.
"""

from argparse import Namespace
import atexit
import asyncio
import gc
import os

import ray
import torch
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import FlexibleArgumentParser


@ray.remote(num_gpus=None, num_cpus=None)
class RayVLLMWorker:
    """A Ray actor that runs a vLLM server.

    Each worker runs an independent vLLM server in a separate thread that can handle
    inference requests.
    """

    def __init__(self, host="0.0.0.0", port=8000, **model_kwargs):
        """Initialize the worker and start the vLLM server.

        Args:
            host (str): Host address to bind the server to
            port (int): Port to run the server on
            **model_kwargs: Additional arguments passed to the vLLM model
        """
        self.model_kwargs = model_kwargs
        self.host = host
        self.port = port

        # Initialize server args
        # Create parser with all AsyncEngineArgs defaults
        openai_parser = make_arg_parser(FlexibleArgumentParser())
        openai_args = openai_parser.parse_args([])
        # Override with host and port
        openai_args.host = self.host
        openai_args.port = self.port
        # Override with all model kwargs
        for key, value in self.model_kwargs.items():
            setattr(openai_args, key, value)

        self.server_args = openai_args

        # Start server in a separate thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Run the server directly in this thread
        self.loop.run_until_complete(run_server(self.server_args))

    def get_config(self):
        """Get the model configuration for this worker."""
        return self.model_kwargs

    def shutdown(self):
        """Shutdown the worker and clean up resources."""
        # The server will automatically shut down when the worker is killed
        gc.collect()
        torch.cuda.empty_cache()


class DistributedVLLM:
    """Manages a distributed system of vLLM workers using Ray.

    This class handles creating and managing multiple vLLM workers across a Ray cluster,
    enabling distributed inference.
    """

    WORKER_NAME_TEMPLATE = "persistent_vllm_worker_{}"
    NAMESPACE = "vllm_workers"

    def __init__(
        self,
        num_workers: int,
        tensor_parallel_size: int = 1,
        persist=True,
        **model_kwargs,
    ):
        """Initialize the distributed system.

        Args:
            num_workers (int): Number of vLLM workers to create
            tensor_parallel_size (int): Number of GPUs to use per worker for tensor parallelism
            kill (bool): If True, will shutdown all workers and exit.
            **model_kwargs: Additional arguments passed to each vLLM model
        """
        try:
            ray.init(address="auto",)
            print("Ray cluster is already initialized.")
        except:
            print("No existing Ray cluster found. Starting a new one...")
            # If ray cluster is not available, start a new clusters.
            total_gpus = num_workers * tensor_parallel_size
            # Get total number of cpus on machine
            total_cpus = os.cpu_count()
            ray.init(
                num_gpus=total_gpus,
                num_cpus=total_cpus - 1,
                namespace=self.NAMESPACE,
                ignore_reinit_error=True,
            )

        self.persist = persist
        # Register shutdown handler
        atexit.register(self.shutdown)

        self.workers = []
        for i in range(num_workers):
            worker_name = self.WORKER_NAME_TEMPLATE.format(i)
            try:
                # Try to get existing worker
                worker = ray.get_actor(worker_name, namespace=self.NAMESPACE)
            except ValueError as e:
                # Worker doesn't exist, create new one
                print(f"Worker {worker_name} doesn't exist, create new one")
                worker = self._create_worker(
                    worker_name, tensor_parallel_size, model_kwargs
                )
            self.workers.append(worker)

    def _create_worker(self, name, tensor_parallel_size, model_kwargs):
        """Create a new Ray vLLM worker.

        Args:
            name (str): Name for the Ray actor
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            model_kwargs (dict): Arguments to pass to the vLLM model

        Returns:
            RayVLLMWorker: A reference to the created worker
        """
        model_kwargs["tensor_parallel_size"] = tensor_parallel_size
        # Avoid conflict with DistributedVLLM using Ray.
        model_kwargs["worker_use_ray"] = False
        try:
            return RayVLLMWorker.options(
                name=name,
                lifetime="detached",  # This makes the actor persist
                num_gpus=tensor_parallel_size,
                num_cpus=tensor_parallel_size * 8,
            ).remote(**model_kwargs)
        except Exception as e:
            print(f"Error creating worker {name}: {str(e)}")
            raise

    def shutdown(self):
        """
        Shutdown the distributed system
        """
        if not self.persist:  # Only shutdown if not persisting or explicitly killing
            print("Shutting down workers...")
            for worker in self.workers:
                try:
                    ray.get(worker.shutdown.remote())
                except:
                    pass  # Ignore errors during shutdown
            ray.shutdown()
        else:
            print("Keeping workers alive (persist=True)")


if __name__ == "__main__":
    distributed_vllm = DistributedVLLM(
        num_workers=1, tensor_parallel_size=4, persist=True, model="Qwen/QwQ-32B-Preview"
    )
