"""
Distributed inference using SGLang.

This module provides distributed inference capabilities by running multiple SGLang servers
across workers. Each worker runs an independent SGLang server that can handle requests
in parallel.

The main components are:
- RaySGLangWorker: Ray actor that runs a single SGLang server
- RaySGLangRouter: Ray actor that routes requests to workers
- DistributedSGLang: Main class that manages the distributed system
"""
import argparse
import dataclasses
import multiprocessing as mp
import os
import requests
from typing import Any, Dict, List

import ray
from setproctitle import setproctitle
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang_router.launch_router import RouterArgs, launch_router

from rllm.globals import BASE_SAMPLER_PORT
from rllm.sampler.utils import (
    check_server_health,
    kill_process_and_children,
    wait_for_server,
)

# Initialize multiprocessing start method if not already set
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

# Port used by the SGLang router
SGLANG_ROUTER_PORT = 30000


def _fetch_default_sglang_args(model: str, tensor_parallel_size: int):
    """
    Create default arguments for SGLang server and router.

    Args:
        model: Model path/name to use
        tensor_parallel_size: Number of GPUs to use per model replica

    Returns:
        Tuple of (ServerArgs, RouterArgs) with default values set
    """
    # Create empty args namespace to avoid requiring command line args
    args = argparse.Namespace()
    # Populate all ServerArgs fields with defaults
    for field in dataclasses.fields(ServerArgs):
        if not hasattr(args, field.name):
            setattr(args, field.name, field.default)

    for field in dataclasses.fields(RouterArgs):
        if not hasattr(args, field.name):
            setattr(args, field.name, field.default)

    args.tensor_parallel_size = tensor_parallel_size
    args.data_parallel_size = 1
    args.expert_parallel_size = 1
    args.model_path = model
    args.exclude_host_port = True

    server_args = ServerArgs.from_cli_args(args)
    router_args = RouterArgs.from_cli_args(args)
    return server_args, router_args


@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangWorker:
    """
    A Ray actor that runs a SGLang server.
    
    Each worker runs an independent SGLang server that can handle inference requests.
    Workers are assigned their own GPU resources and run independently.
    """
    
    def __init__(self, port: int, **model_kwargs):
        """
        Initialize the worker and prepare SGLang server configuration.
        
        Args:
            port: Port to run the server on
            **model_kwargs: Additional arguments for model configuration
        """
        self.port = port
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        self.tensor_parallel_size = model_kwargs.get("tensor_parallel_size") or model_kwargs.get("tp_size", 1)
        
        # Get server args
        self.server_args, _ = _fetch_default_sglang_args(
            model=self.model, 
            tensor_parallel_size=self.tensor_parallel_size
        )
        self.server_args.port = self.port
        # Each worker starts at GPU 0 since they're independent
        self.server_args.base_gpu_id = 0
        self.server_args.dp_size = 1

    def start_server(self):
        """Start the SGLang server process."""
        try:
            # Try to set process group and title
            os.setpgrp()
            try:
                from setproctitle import setproctitle
                setproctitle("sglang::server")
            except ImportError:
                print("Warning: setproctitle not available, skipping process title setting")
        except Exception as e:
            print(f"Warning: Could not set process group: {e}")
            
        os.environ["SGLANG_DP_RANK"] = '0'
        launch_server(self.server_args)
    
    def get_pid(self):
        """Get process ID of the server."""
        return os.getpid()


@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangRouter:
    """
    A Ray actor that runs the SGLang router.
    
    The router distributes requests across available workers and aggregates responses.
    """
    
    def __init__(self, worker_ports: List[int], **model_kwargs):
        """
        Initialize the router configuration.
        
        Args:
            worker_ports: List of ports where workers are running
            **model_kwargs: Additional arguments for model configuration
        """
        self.worker_ports = worker_ports
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        self.tensor_parallel_size = model_kwargs.get("tensor_parallel_size") or model_kwargs.get("tp_size", 1)
        
        # Get router args
        _, self.router_args = _fetch_default_sglang_args(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size
        )
        self.router_args.worker_urls = [f"http://0.0.0.0:{port}" for port in self.worker_ports]
        self.router_args.port = SGLANG_ROUTER_PORT

    def start_router(self):
        """Start the router process."""
        launch_router(self.router_args)
    
    def get_pid(self):
        """Get process ID of the router."""
        return os.getpid()


class DistributedSGLang:
    """
    Manages a distributed system of SGLang workers using Ray.
    
    This class handles:
    - Starting/stopping Ray cluster
    - Managing worker processes
    - Routing inference requests
    - Resource cleanup
    """

    WORKER_NAME_TEMPLATE = "persistent_sglang_worker_{}"
    ROUTER_NAME = "persistent_sglang_router"
    NAMESPACE = "sglang_workers"

    def __init__(self, num_workers: int = 1, **model_kwargs):
        """
        Initialize the distributed system.

        Args:
            num_workers: Number of worker processes to launch
            **model_kwargs: Model configuration parameters
        """
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        try:
            ray.init(address="auto", namespace=self.NAMESPACE)
            print("Ray cluster is already initialized.")
        except:
            print("Starting new Ray cluster...")
            tensor_parallel_size = model_kwargs.get("tensor_parallel_size", 1)
            total_gpus = num_workers * tensor_parallel_size
            total_cpus = os.cpu_count()
            os.system(f"ray start --head --num-cpus={total_cpus-1} --num-gpus={total_gpus}")
            ray.init(address="auto", namespace=self.NAMESPACE)
        self.num_workers = num_workers
        self.model_kwargs = model_kwargs
        # Fetch SGLang workers
        self._get_or_launch_workers()
        
        # Fetch SGLang router
        self._get_or_launch_router()
        print("All SGLang servers and router have started successfully.")

    def _get_or_launch_workers(self):
        """Launch SGLang workers or connect to existing ones."""
        self.worker_pids = []
        worker_refs = []
        new_worker_refs = []
        self.worker_ports = [BASE_SAMPLER_PORT + i for i in range(self.num_workers)]
        for i in range(self.num_workers):
            worker_name = self.WORKER_NAME_TEMPLATE.format(i)
            try:
                worker = ray.get_actor(worker_name, namespace=self.NAMESPACE)
                print(f"Found existing worker: {worker_name}")
            except ValueError:
                print(f"Creating new worker: {worker_name}")
                worker = RaySGLangWorker.options(
                    name=worker_name,
                    lifetime="detached", 
                    namespace=self.NAMESPACE,
                    num_gpus=self.model_kwargs.get("tensor_parallel_size", 1),
                    num_cpus=8,
                ).remote(
                    port=self.worker_ports[i],
                    **self.model_kwargs
                )
                new_worker_refs.append(worker)
            worker_refs.append(worker)
            actor_info = ray._private.state.actors()[worker._actor_id.hex()]
            worker_pid = actor_info['Pid']
            self.worker_pids.append(worker_pid)

        # Start new workers if any
        if new_worker_refs:
            [w.start_server.remote() for w in new_worker_refs]
            # Wait for workers to be healthy
            for port in self.worker_ports:
                wait_for_server(port)
        
        self.workers = worker_refs
    
    def _get_or_launch_router(self):
        """Launch router or connect to existing one."""
        self.router_pid = None
        try:
            self.router = ray.get_actor(self.ROUTER_NAME, namespace=self.NAMESPACE)
            print("Found existing router")
            # Check if router is healthy
            if not check_server_health(SGLANG_ROUTER_PORT):
                raise ValueError("Router not healthy")
        except ValueError:
            print("Creating new router")
            self.router = RaySGLangRouter.options(
                name=self.ROUTER_NAME,
                lifetime="detached",
                namespace=self.NAMESPACE,
                num_cpus=8,
            ).remote(self.worker_ports, **self.model_kwargs)
            
            actor_info = ray._private.state.actors()[self.router._actor_id.hex()]
            self.router_pid = actor_info['Pid']
            self.router.start_router.remote()
            wait_for_server(SGLANG_ROUTER_PORT)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request through the router.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the chat completion API
            
        Returns:
            Dict containing the API response
            
        Raises:
            Exception: If the request fails
        """     
        endpoint = f"http://localhost:{SGLANG_ROUTER_PORT}/v1/chat/completions"
        
        payload = {
            "messages": messages,
            "model": self.model,
            **kwargs
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise e

    def shutdown(self):
        """
        Shutdown the distributed system and clean up resources.
        
        Terminates all worker processes, router process, and Ray cluster.
        """
        print("Shutting down SGLang servers and router...")
        
        shutdown_results = []
        
        # Shutdown workers using stored PIDs
        if hasattr(self, 'worker_pids'):
            for worker_pid in self.worker_pids:
                success = kill_process_and_children(worker_pid)
                shutdown_results.append(success)
                if success:
                    print(f"Successfully terminated worker process {worker_pid}")
                else:
                    print(f"Failed to terminate worker process {worker_pid}")

        # Shutdown router using stored PID
        if hasattr(self, 'router_pid') and self.router_pid:
            success = kill_process_and_children(self.router_pid)
            if success:
                print(f"Successfully terminated router process {self.router_pid}")
            else:
                print(f"Failed to terminate router process {self.router_pid}")
            shutdown_results.append(success)

        # Stop Ray
        try:
            ray.shutdown()
            os.system("ray stop")
        except Exception as e:
            print(f"Error stopping Ray: {e}")
            shutdown_results.append(False)

        if all(shutdown_results):
            print("All processes terminated successfully")
        else:
            print("Some processes may not have terminated properly")


if __name__ == "__main__":
    # Testing DistributedVLLM
    distributed_sampler = DistributedSGLang(
        num_workers=2,
        tensor_parallel_size=2,
        model="Qwen/QwQ-32B-Preview"
    )
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of France?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    distributed_sampler.shutdown()
