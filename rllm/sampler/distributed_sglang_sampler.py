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
import gc
import multiprocessing as mp
import os
import threading
from typing import Any, Dict, List, Tuple

import ray
import requests
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang_router.launch_router import RouterArgs
from transformers import AutoTokenizer
from rllm.sampler import Sample, SampleBatch
from rllm.sampler.utils import convert_openai_response_to_samples
from openai import OpenAI

from rllm.globals import BASE_SAMPLER_PORT
from rllm.sampler.utils import (
    kill_process_and_children,
    wait_for_server,
)

# Initialize multiprocessing start method if not already set
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

# Port used by the SGLang router
SGLANG_ROUTER_PORT = 8000


def _fetch_default_sglang_args(model: str, tensor_parallel_size: int) -> Tuple[ServerArgs, RouterArgs]:
    """Create default arguments for SGLang server and router.

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


def _launch_sglang_server(server_args: ServerArgs) -> None:
    """Target function for our child process.

    This is where we call the blocking launch_server(server_args).
    Because this runs in a separate process, it doesn't block the Ray actor's thread.
    """
    os.environ["SGLANG_DP_RANK"] = "0"

    # Optionally set process name.
    try:
        from setproctitle import setproctitle
        setproctitle("sglang_server_subprocess")
    except ImportError:
        pass

    print(f"[Child Process] Launching SGLang server on port {server_args.port}")
    # This call never returns until the process is killed/shutdown.
    launch_server(server_args)


@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangWorker:
    """A Ray actor that runs a SGLang server.

    Each worker runs an independent SGLang server that can handle inference requests.
    Workers are assigned their own GPU resources and run independently.
    """

    def __init__(self, port: int, **model_kwargs) -> None:
        """Initialize the worker and prepare SGLang server configuration.

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

        # We'll store a reference to the child Process so we can shut it down later.
        self.server_process = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Initialize OpenAI client
        self.client = OpenAI(base_url=f"http://0.0.0.0:{self.port}/v1", api_key="not-needed")

    def start_server(self) -> str:
        """Spawn a child process that runs launch_server(self.server_args).

        Returns:
            Status message indicating if server started or was already running
        """
        if self.server_process is not None:
            return f"SGLang server already running on port {self.port}"

        # Create the child process
        self.server_process = mp.Process(
            target=_launch_sglang_server,
            args=(self.server_args,),
            daemon=False  # Usually want the server to keep running until we explicitly stop it
        )
        # Start the process
        self.server_process.start()
        # (Optional) wait for server to become healthy
        wait_for_server(self.port)
        return f"Started server on port {self.port}"

    def shutdown(self) -> str:
        """Terminate the SGLang server process (if running).

        Returns:
            Status message indicating if server was terminated
        """
        if self.server_process is None:
            return "No server process to terminate."

        pid = self.server_process.pid
        kill_process_and_children(pid)
        self.server_process = None
        return f"Server on port {self.port} has been shut down."

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> SampleBatch:
        """Send a chat completion request to this worker.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the chat completion API
            
        Returns:
            SampleBatch containing the response and metrics
        """
        try:
            chat_response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                logprobs=True,
                top_logprobs=1,
                **kwargs
            )
        except Exception as e:
            return {'Error': str(e)}

        samples = convert_openai_response_to_samples(chat_response)
        sample_lengths = []
        for sample in samples:
            sample.tokens = self.tokenizer.encode(sample.response)
            sample_lengths.append(len(sample.tokens))
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_tokens = self.tokenizer.encode(prompt)
        num_prompt_tokens = len(prompt_tokens)
        
        return SampleBatch(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            samples=samples,
            metrics={
                "prompt_tokens": num_prompt_tokens,
                "completion_tokens": sum(sample_lengths),
                "sample_tokens": sample_lengths,
            })


# Deprecated.
@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangRouter:
    """A Ray actor that runs the SGLang router.

    The router distributes requests across available workers and aggregates responses.
    """

    def __init__(self, worker_ports: List[int], **model_kwargs) -> None:
        """Initialize the router configuration.

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
        self.router_process = None

    def start_router(self) -> str:
        """Spawn a child process that runs launch_server(self.server_args).

        Returns:
            Status message indicating if router started or was already running
        """
        if self.router_process is not None:
            return f"SGLang server already running on port {self.port}"

        # Create the child process
        self.router_process = mp.Process(
            target=_launch_sglang_server,
            args=(self.router_args,),
            daemon=False  # Usually want the server to keep running until we explicitly stop it
        )
        # Start the process
        self.router_process.start()
        # (Optional) wait for server to become healthy
        wait_for_server(SGLANG_ROUTER_PORT)
        return f"Started router on port {SGLANG_ROUTER_PORT}"

    def shutdown(self) -> str:
        """Terminate the SGLang router process (if running).

        Returns:
            Status message indicating if router was terminated
        """
        if self.router_process is None:
            return "No router process to terminate."

        pid = self.router_process.pid
        kill_process_and_children(pid)
        self.router_process = None
        return "Router has been shut down."


class DistributedSGLang:
    """Manages a distributed system of SGLang workers using Ray.

    This class handles:
    - Starting/stopping Ray cluster
    - Managing worker processes
    - Routing inference requests
    - Resource cleanup
    """

    WORKER_NAME_TEMPLATE = "persistent_sglang_worker_{}"
    ROUTER_NAME = "persistent_sglang_router"
    NAMESPACE = "sglang_workers"

    def __init__(self, num_workers: int = 1, **model_kwargs) -> None:
        """Initialize the distributed system.

        Args:
            num_workers: Number of worker processes to launch
            **model_kwargs: Model configuration parameters
        """
        self._worker_lock = threading.Lock()
        self.model = model_kwargs.get("model_path") or model_kwargs.get("model", "facebook/opt-125m")
        try:
            ray.init(address="auto", namespace=self.NAMESPACE)
            print("Ray cluster is already initialized.")
        except ConnectionError:
            print("Starting new Ray cluster...")
            tensor_parallel_size = model_kwargs.get("tensor_parallel_size", 1)
            total_gpus = num_workers * tensor_parallel_size
            total_cpus = os.cpu_count()
            os.system(f"ray start --head --num-cpus={total_cpus-1} --num-gpus={total_gpus}")
            ray.init(address="auto", namespace=self.NAMESPACE)
        self.num_workers = num_workers
        self.model_kwargs = model_kwargs
        self.active_requests = {}  # Dict to track active requests per worker
        # Fetch SGLang workers
        self._get_or_launch_workers()

        # Fetch SGLang router
        # self._get_or_launch_router()
        print("All SGLang servers and router have started successfully.")

    def _get_or_launch_workers(self) -> None:
        """Launch SGLang workers or connect to existing ones."""
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
                    max_concurrency=256,
                ).remote(
                    port=self.worker_ports[i],
                    **self.model_kwargs
                )
                new_worker_refs.append(worker)
            worker_refs.append(worker)
            self.active_requests[i] = 0

        # Start new workers if any
        if new_worker_refs:
            ray.get([w.start_server.remote() for w in new_worker_refs])

        self.workers = worker_refs

    def _get_least_busy_worker(self) -> Tuple[str, int]:
        """Get the URL of the worker with the least number of active requests.

        Returns:
            Tuple of (worker URL, worker index)
        """
        with self._worker_lock:
            least_busy_idx = min(self.active_requests, key=self.active_requests.get)
            port = BASE_SAMPLER_PORT + least_busy_idx
            self.active_requests[least_busy_idx] += 1
            return f"http://0.0.0.0:{port}", least_busy_idx

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send a chat completion request to the least busy worker.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the chat completion API
            
        Returns:
            SampleBatch containing the response and metrics
        """
        url, worker_idx = self._get_least_busy_worker()
        print(f"Using worker {worker_idx} at {url}; {self.active_requests}")
        
        try:
            sample_batch = ray.get(self.workers[worker_idx].chat_completion.remote(messages=messages, **kwargs))
            with self._worker_lock:
                self.active_requests[worker_idx] -= 1
            return sample_batch
        except Exception as e:
            with self._worker_lock:
                self.active_requests[worker_idx] -= 1
            raise e

    def shutdown(self) -> None:
        """Shutdown the distributed system and clean up resources.

        Args:
            shutdown_ray: Whether to also shutdown the Ray cluster

        Terminates all worker processes, router process, and optionally Ray cluster.
        """
        print("Shutting down SGLang servers and router...")

        for worker in self.workers:
            # Kill worker 
            ray.get(worker.shutdown.remote())
            ray.kill(worker)

    def restart(self):
        """Restart all workers."""
        self.shutdown()
        self._get_or_launch_workers()


if __name__ == "__main__":
    # Testing DistributedSGLang
    distributed_sampler = DistributedSGLang(
        num_workers=2,
        tensor_parallel_size=2,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    import pdb; pdb.set_trace()
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of France?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_sampler.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    distributed_sampler.shutdown(shutdown_ray=False)
