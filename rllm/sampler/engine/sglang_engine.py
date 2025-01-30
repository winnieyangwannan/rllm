"""SGLang Engine Implementation

This module implements the SGLang engine for distributed model inference using Ray actors.
It provides worker and router classes that manage SGLang server processes for handling
inference requests across multiple GPUs.

The main components are:
- RaySGLangWorker: A Ray actor that runs an individual SGLang server process and handles
  inference requests. Each worker manages its own GPU resources.
- RaySGLangRouter (Deprecated): A Ray actor that routes requests across multiple workers.

The module also includes helper functions for:
- Creating default SGLang server/router configurations
- Launching server processes
- Managing process lifecycle
- Converting between different response formats

Key Features:
- Distributed inference across multiple GPUs using tensor parallelism
- Chat completion API compatible with OpenAI interface
- Automatic server process management and cleanup
- Tokenization and response processing
"""
from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import signal
from typing import Any, Dict, List, Tuple

from openai import OpenAI
import ray
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang_router.launch_router import RouterArgs
from transformers import AutoTokenizer

from rllm.globals import BASE_SAMPLER_PORT
from rllm.sampler.sampler_types import SampleBatch
from rllm.sampler.utils import (
    convert_openai_response_to_samples,
    kill_process_and_children,
    wait_for_server,
)

# Initialize multiprocessing start method if not already set
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

# Port used by the SGLang router
SGLANG_ROUTER_PORT = 30000


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
    """Launch SGLang server in a child process.

    This function runs in a separate process and starts the SGLang server with the given
    arguments. The server runs until explicitly terminated.

    Args:
        server_args: Configuration arguments for the SGLang server
    """
    os.environ["SGLANG_DP_RANK"] = "0"

    # Optionally set process name
    try:
        from setproctitle import setproctitle
        setproctitle("sglang_server_subprocess")
    except ImportError:
        pass

    print(f"[Child Process] Launching SGLang server on port {server_args.port}")
    # This call blocks until the process is killed/shutdown
    launch_server(server_args)


@ray.remote(num_gpus=None, num_cpus=None)
class RaySGLangWorker:
    """A Ray actor that runs a SGLang server.

    Each worker runs an independent SGLang server that can handle inference requests.
    Workers are assigned their own GPU resources and run independently.

    Attributes:
        port: Port number the server listens on
    """

    def __init__(self, port: int, **model_kwargs) -> None:
        """Initialize the worker and prepare SGLang server configuration.

        Args:
            port: Port to run the server on
            **model_kwargs: Additional arguments for model configuration including:
                - model_path/model: Model name or path
                - tensor_parallel_size/tp_size: Number of GPUs for tensor parallelism
        """
        if os.environ.get("IGNORE_SIGINT", "") == "1":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
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

        self.server_process = None
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Initialize OpenAI client
        self.client = OpenAI(base_url=f"http://0.0.0.0:{self.port}/v1", api_key="not-needed", timeout=int(1e9))
    
    def wait_for_server(self) -> None:
        """Wait for the server to become healthy."""
        wait_for_server(self.port)
    
    def start_server(self) -> str:
        """Start the SGLang server in a subprocess.

        Returns:
            Status message indicating if server started or was already running
        """
        if self.server_process is not None:
            return f"SGLang server already running on port {self.port}"

        # Create the child process
        self.server_process = mp.Process(
            target=_launch_sglang_server,
            args=(self.server_args,),
            daemon=False  # Keep server running until explicitly stopped
        )
        # Start the process
        self.server_process.start()

    def wait_for_server(self) -> None:
        """Wait for the server to become healthy."""
        wait_for_server(self.port)
        print(f"Started server on port {self.port}")

    def shutdown(self) -> str:
        """Terminate the SGLang server process.

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
        num_retries = 10
        for retry in range(num_retries):
            try:
                chat_response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    logprobs=True,
                    top_logprobs=1,
                    **kwargs
                )
                break
            except Exception as exc:
                import traceback
                traceback.print_exc()
                if retry == num_retries - 1:
                    return {'Error': str(exc)}

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
    This class is deprecated and will be removed in a future version.

    Attributes:
        worker_ports: List of ports where workers are running
        model: Name or path of the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        router_args: Configuration for the SGLang router
        router_process: Handle to the router subprocess
    """

    def __init__(self, worker_ports: List[int], **model_kwargs) -> None:
        """Initialize the router configuration.

        Args:
            worker_ports: List of ports where workers are running
            **model_kwargs: Additional arguments for model configuration including:
                - model_path/model: Model name or path
                - tensor_parallel_size/tp_size: Number of GPUs for tensor parallelism
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
        """Start the SGLang router in a subprocess.

        Returns:
            Status message indicating if router started or was already running
        """
        if self.router_process is not None:
            return f"SGLang server already running on port {self.port}"

        # Create the child process
        self.router_process = mp.Process(
            target=_launch_sglang_server,
            args=(self.router_args,),
            daemon=False  # Keep router running until explicitly stopped
        )
        # Start the process
        self.router_process.start()
        # Wait for router to become healthy
        wait_for_server(SGLANG_ROUTER_PORT)
        return f"Started router on port {SGLANG_ROUTER_PORT}"

    def shutdown(self) -> str:
        """Terminate the SGLang router process.

        Returns:
            Status message indicating if router was terminated
        """
        if self.router_process is None:
            return "No router process to terminate."

        pid = self.router_process.pid
        kill_process_and_children(pid)
        self.router_process = None
        return "Router has been shut down."