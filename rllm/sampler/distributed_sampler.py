"""
Distributed inference using vLLM and Ray.

This module provides distributed inference capabilities by running multiple vLLM servers
across Ray workers. Each worker runs an independent vLLM server that can handle requests
in parallel.
"""

import atexit
import asyncio
import gc
import os
import requests
import time

import ray
import torch

from vllm.entrypoints.openai import api_server as vllm_api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import FlexibleArgumentParser

from rllm.globals import BASE_VLLM_PORT
# Monkey patch the vLLM server to disable signal handlers
from rllm.sampler.monkey_patch import run_server
vllm_api_server.run_server = run_server

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
        openai_args.disable_signal_handlers = True
        # Override with all model kwargs
        for key, value in self.model_kwargs.items():
            setattr(openai_args, key, value)

        self.server_args = openai_args

    def get_pid(self):
        """Get the process ID of this worker."""
        return os.getpid()

    def start_server(self):
        """This runs in a dedicated thread, so we can block here safely."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server(self.server_args,))
        loop.close()
    
    def wait_for_server(self, timeout=600, interval=5):
        """Wait for the server to be ready by checking the health endpoint.
        
        Args:
            timeout (int): Maximum time to wait in seconds
            interval (int): Time between health checks in seconds
        
        Raises:
            TimeoutError: If server doesn't become ready within timeout period
        """
        health_url = f"http://{self.host}:{self.port}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=interval)
                if response.status_code == 200:
                    print(f"Server ready on {self.host}:{self.port}")
                    return
            except requests.exceptions.RequestException as e:
                pass
            except Exception as e:
                pass
            time.sleep(interval)      
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

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
        num_workers: int = 1,
        **model_kwargs,
    ):
        """Initialize the distributed system.

        Args:
            num_workers (int): Number of vLLM workers to create
            **model_kwargs: Additional arguments passed to each vLLM model
        """
        try:
            ray.init(address="auto", namespace=self.NAMESPACE)
            print("Ray cluster is already initialized.")
        except:
            print("No existing Ray cluster found. Starting a new one...")
            # If ray cluster is not available, start a new clusters.
            tensor_parallel_size = model_kwargs.get("tensor_parallel_size", 1)
            total_gpus = num_workers * tensor_parallel_size
            total_actual_gpus = torch.cuda.device_count()
            if total_gpus > total_actual_gpus:
                raise ValueError(f"Requested {total_gpus} GPUs, but only {total_actual_gpus} are available.")
            # Get total number of cpus on machine
            total_cpus = os.cpu_count()
            os.system(f"ray start --head --num-cpus={total_cpus-1} --num-gpus={total_gpus}")
            ray.init(address="auto", namespace=self.NAMESPACE)

        # ---- 1) Create or get existing actors (no blocking) ----
        actor_refs = []
        for i in range(num_workers):
            worker_name = self.WORKER_NAME_TEMPLATE.format(i)
            try:
                # Try to get an existing named actor
                worker = ray.get_actor(worker_name, namespace=self.NAMESPACE)
                print(f"Found existing worker: {worker_name}")
            except ValueError:
                # If it doesn't exist, create a new one
                print(f"Worker {worker_name} doesn't exist, creating a new one.")
                worker = self._create_worker(
                    name=worker_name,
                    model_kwargs=model_kwargs,
                    port=BASE_VLLM_PORT + i
                )
            actor_refs.append(worker)
        
        
        # ---- 2) Start all servers in parallel (non-blocking calls) ----
        [actor.start_server.remote() for actor in actor_refs]

        # ---- 3) Wait for all servers to report they're healthy (still parallel) ----
        wait_futures = [actor.wait_for_server.remote() for actor in actor_refs]

        # Ray will block until all wait_for_server calls are finished
        # (i.e., all servers are healthy). This is truly parallel
        # because each actor is running in its own process.
        ray.get(wait_futures)
        
        print("All vLLM servers have started successfully.")
        self.workers = actor_refs

    def _create_worker(self, name, model_kwargs, port=BASE_VLLM_PORT):
        """Create a new Ray vLLM worker.

        Args:
            name (str): Name for the Ray actor
            model_kwargs (dict): Arguments to pass to the vLLM model
            port (int): Port number for the worker's server

        Returns:
            RayVLLMWorker: A reference to the created worker
        """
        # Avoid conflict with DistributedVLLM using Ray.
        model_kwargs["worker_use_ray"] = False
        model_kwargs["engine_use_ray"] = False
        tensor_parallel_size = model_kwargs.get("tensor_parallel_size", 1)
        try:
            ray_actor =  RayVLLMWorker.options(
                name=name,
                lifetime="detached",  # This makes the actor persist
                namespace=self.NAMESPACE,
                num_gpus=tensor_parallel_size,
                num_cpus=tensor_parallel_size * 8,
                max_concurrency=2,
            ).remote(port=port, **model_kwargs)
            return ray_actor
        except Exception as e:
            print(f"Error creating worker {name}: {str(e)}")
            raise

    def shutdown(self):
        """
        Shutdown the distributed system
        """
        print("Shutting down Ray and vLLM workers...")
        for worker in self.workers:
            try:
                # Get worker PID
                worker_pid = ray.get(worker.get_pid.remote())
                # Kill the worker and all its descendant processes
                os.system(f"pkill -9 -g $(ps -o pgid= {worker_pid} | tr -d ' ')")
                ray.get(worker.shutdown.remote())
                ray.kill(worker)
            except Exception as e:
                print(f"Error shutting down worker {worker}: {str(e)}")
                pass  # Ignore errors during shutdown
        os.system('ray stop --force')


if __name__ == "__main__":
    distributed_vllm = DistributedVLLM(
        num_workers=4, tensor_parallel_size=2, model="Qwen/QwQ-32B-Preview"
    )
    distributed_vllm.shutdown()
