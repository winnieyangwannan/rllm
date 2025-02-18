"""vLLM Engine Implementation

This module provides distributed inference capabilities by running multiple vLLM servers
across workers. Each worker runs an independent vLLM server that can handle requests
in parallel.

Key Components:
- RayVLLMWorker: Ray actor that runs a single vLLM server instance
- _launch_vllm_server: Helper function to launch the vLLM server process

Features:
- Distributed inference across multiple GPUs using tensor parallelism
- Continuous batching for efficient throughput
- Prefix caching to avoid redundant computation
- PagedAttention for memory-efficient inference
- OpenAI-compatible chat completion API
"""
import asyncio
import gc
import multiprocessing as mp
import os
import signal
import traceback
from typing import Any, Dict, List

from openai import AsyncOpenAI
import ray
import torch
from transformers import AutoTokenizer
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import FlexibleArgumentParser

from rllm.sampler.sampler_types import SampleBatch
from rllm.sampler.utils import (
    convert_openai_response_to_samples,
    kill_process_and_children,
    wait_for_server,
)

# Optimized configuration for vLLM server
OPTIMIZED_VLLM_CONFIG = {
    "max_num_seqs": 1024,
    "max_num_batched_tokens": 2**16,
    "enable_chunked_prefill": True,
#    "num_scheduler_steps": 40,
    "enable_prefix_caching": True,
    "dtype": "bfloat16",  # Add this line to set default dtype
}


def _launch_vllm_server(args: Any) -> None:
    """Launch a vLLM server in a child process.

    This function runs in a separate process and starts the vLLM server with the given
    arguments. It sets up an asyncio event loop to run the server.

    Args:
        args: Configuration arguments for the vLLM server
    """
    print(f"[Child Process] Launching vLLM server on port {args.port}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server(args))
    loop.close()


@ray.remote(num_gpus=None, num_cpus=None)
class RayVLLMWorker:
    """A Ray actor that runs a vLLM server.

    Each worker runs an independent vLLM server in a separate process that can handle
    inference requests. The worker manages server lifecycle and provides a chat
    completion interface compatible with OpenAI's API.

    Attributes:
        port: Port number the server listens on
        model: Name or path of the model to load
        server_args: Configuration for the vLLM server
        server_process: Reference to the server process
        oai_client: OpenAI client for making requests
        tokenizer: Tokenizer for the loaded model
    """

    def __init__(self, port: int, **model_kwargs) -> None:
        """Initialize the worker and prepare vLLM server configuration.

        Args:
            port: Port to run the server on
            **model_kwargs: Additional arguments for model configuration including:
                - model: Model name/path
                - tensor_parallel_size: Number of GPUs for tensor parallelism
                - Other vLLM engine parameters
        """
        if os.environ.get("IGNORE_SIGINT", "") == "1":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.port = port
        self.model = model_kwargs.get("model", "facebook/opt-125m")
        
        # Initialize server args
        openai_parser = make_arg_parser(FlexibleArgumentParser())
        self.server_args = openai_parser.parse_args(["--enable-auto-tool-choice", "--tool-call-parser", "hermes"])
        
        # Configure server args
        self.server_args.host = "0.0.0.0"
        self.server_args.port = self.port
        self.server_args.disable_signal_handlers = True
        
        # Apply optimized config
        for key, value in OPTIMIZED_VLLM_CONFIG.items():
            setattr(self.server_args, key, value)
        
        # Apply model kwargs
        for key, value in model_kwargs.items():
            setattr(self.server_args, key, value)

        self.server_process = None
        # self.oai_client = OpenAI(api_key='EMPTY', base_url=f"http://0.0.0.0:{self.port}/v1", timeout=int(1e9))
        self.oai_client = AsyncOpenAI(api_key='EMPTY', base_url=f"http://0.0.0.0:{self.port}/v1", timeout=int(1e9))
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def start_server(self) -> str:
        """Spawn a child process that runs the vLLM server.

        Returns:
            Status message indicating if server started or was already running
        """
        if self.server_process is not None:
            return f"vLLM server already running on port {self.port}"

        # Create and start the child process
        self.server_process = mp.Process(
            target=_launch_vllm_server,
            args=(self.server_args,),
            daemon=False
        )
        self.server_process.start()
    
    def wait_for_server(self) -> None:
        """Wait for the server to become healthy."""
        wait_for_server(self.port)
        print(f"Started server on port {self.port}")
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> SampleBatch:
        """Send an async chat completion request to this worker.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the chat completion API
            
        Returns:
            SampleBatch containing the response and metrics
        """
        try:
            chat_response = await self.oai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                logprobs=True,
                top_logprobs=1,
                extra_body={
                    "top_k": -1,
                    "min_p": 0.0,
                },
                **kwargs
            )
        except Exception as e:
            traceback.print_exc()
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

    def shutdown(self) -> str:
        """Terminate the vLLM server process and clean up resources.

        Returns:
            Status message indicating if server was terminated
        """
        if self.server_process is None:
            return "No server process to terminate."

        pid = self.server_process.pid
        kill_process_and_children(pid)
        self.server_process = None
        
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return f"Server on port {self.port} has been shut down."
