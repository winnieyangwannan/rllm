"""
Distributed inference using SGLang.

This module provides distributed inference capabilities by running multiple SGLang servers
across workers. Each worker runs an independent SGLang server that can handle requests
in parallel.
"""
import argparse
import psutil
import asyncio
import gc
import os
import requests
import time
import threading
from typing import List, Dict, Any
import multiprocessing as mp
import logging
import random
import signal
from setproctitle import setproctitle
import copy
import multiprocessing as mp
import ray
import torch
import dataclasses
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang_router.launch_router import RouterArgs, launch_router

from rllm.globals import BASE_SAMPLER_PORT
from rllm.utils import find_available_ports, is_port_available

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

SGLANG_ROUTER_PORT = 30000

def _kill_process_and_children(pid: int) -> bool:
    """Kill a process and all its children processes.
    
    Args:
        pid: Process ID to kill
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            
        # Kill parent
        parent.kill()
        
        # Wait for processes to terminate
        psutil.wait_procs(children + [parent], timeout=3)
        
        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def _check_server_health(port: int, timeout: int = 5) -> bool:
    """Check if server at given port is healthy"""
    try:
        response = requests.get(f"http://0.0.0.0:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def _wait_for_server(port: int, timeout: int = 600, interval: int = 5):
    """Wait for server to be healthy"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _check_server_health(port):
            return True
        time.sleep(interval)
    raise TimeoutError(f"Server on port {port} failed to start within {timeout} seconds")

def _fetch_default_sglang_args(model: str, tensor_parallel_size: int):
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
    """A Ray actor that runs a SGLang server.
    
    Each worker runs an independent SGLang server that can handle inference requests.
    """
    
    def __init__(self, port: int, dp_id: int, **model_kwargs):
        """Initialize the worker and start the SGLang server.
        
        Args:
            port (int): Port to run the server on
            dp_id (int): Data parallel ID for this worker, should be 0 since each worker is independent
            **model_kwargs: Additional arguments passed to the model
        """
        self.port = port
        # Each worker should have dp_id=0 since they run independently
        self.dp_id = 0  # Changed from dp_id parameter since each worker is independent
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        self.tensor_parallel_size = model_kwargs.get("tensor_parallel_size") or model_kwargs.get("tp_size", 1)
        
        # Get server args
        self.server_args, _ = _fetch_default_sglang_args(
            model=self.model, 
            tensor_parallel_size=self.tensor_parallel_size
        )
        self.server_args.port = self.port
        # Each worker starts at GPU 0 since they're independent
        self.server_args.base_gpu_id = 0  # Changed from dp_id * tp_size since each worker is independent
        self.server_args.dp_size = 1

    def start_server(self):
        """Start the SGLang server."""
        try:
            # Try to set process group and title if setproctitle is available
            os.setpgrp()
            try:
                from setproctitle import setproctitle
                setproctitle(f"sglang::server")
            except ImportError:
                print("Warning: setproctitle not available, skipping process title setting")
        except Exception as e:
            print(f"Warning: Could not set process group: {e}")
            
        os.environ["SGLANG_DP_RANK"] = str(self.dp_id)  # Will always be 0
        launch_server(self.server_args)
    
    def get_pid(self):
        return os.getpid()


@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangRouter:
    """A Ray actor that runs the SGLang router."""
    
    def __init__(self, worker_ports: List[int], **model_kwargs):
        """Initialize the router.
        
        Args:
            worker_ports: List of ports where workers are running
            **model_kwargs: Additional arguments for the model
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
        """Start the router."""
        launch_router(self.router_args)
    
    def get_pid(self):
        return os.getpid()

class DistributedSGLang:
    """Manages a distributed system of SGLang workers using Ray."""

    WORKER_NAME_TEMPLATE = "persistent_sglang_worker_{}"
    ROUTER_NAME = "persistent_sglang_router"
    NAMESPACE = "sglang_workers"

    def __init__(self, num_workers: int = 1, **model_kwargs):
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
        """Launch SGLang workers and router."""
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
                    dp_id=0,  # Each worker has dp_id=0 since they're independent
                    **self.model_kwargs
                )
                new_worker_refs.append(worker)
            worker_refs.append(worker)
            actor_info = ray._private.state.actors()[worker._actor_id.hex()]
            worker_pid = actor_info['Pid']
            self.worker_pids.append(worker_pid)
            #self.worker_pids.append(ray.get(worker.get_pid.remote()))
            

        # Start new workers if any
        if new_worker_refs:
            [w.start_server.remote() for w in new_worker_refs]
            # Wait for workers to be healthy
            for port in self.worker_ports:
                _wait_for_server(port)
        
        self.workers = worker_refs
    
    def _get_or_launch_router(self):
        # Create or get existing router
        self.router_pid = None
        try:
            self.router = ray.get_actor(self.ROUTER_NAME, namespace=self.NAMESPACE)
            print("Found existing router")
            # Check if router is healthy
            if not _check_server_health(SGLANG_ROUTER_PORT):
                raise ValueError("Router not healthy")
        except ValueError:
            print("Creating new router")
            self.router = RaySGLangRouter.options(
                name=self.ROUTER_NAME,
                lifetime="detached",
                namespace=self.NAMESPACE,
                num_cpus=8,
            ).remote(self.worker_ports, **self.model_kwargs)
            
            #self.router_pid = ray.get(self.router.get_pid.remote())
            actor_info = ray._private.state.actors()[self.router._actor_id.hex()]
            self.router_pid = actor_info['Pid']
            self.router.start_router.remote()
            _wait_for_server(SGLANG_ROUTER_PORT)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send a chat completion request through the router.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the chat completion API
            
        Returns:
            Dict containing the API response
        """     
        if not _check_server_health(SGLANG_ROUTER_PORT):
            raise RuntimeError("Router is not healthy")
            
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
        """Shutdown the distributed system and clean up resources."""
        print("Shutting down SGLang servers and router...")
        
        shutdown_results = []
        
        # Shutdown workers using stored PIDs
        if hasattr(self, 'worker_pids'):
            for worker_pid in self.worker_pids:
                success = _kill_process_and_children(worker_pid)
                shutdown_results.append(success)
                if success:
                    print(f"Successfully terminated worker process {worker_pid}")
                else:
                    print(f"Failed to terminate worker process {worker_pid}")

        # Shutdown router using stored PID
        if hasattr(self, 'router_pid') and self.router_pid:
            success = _kill_process_and_children(self.router_pid)
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
