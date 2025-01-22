"""
Distributed inference using vLLM and Ray.

This module provides distributed inference capabilities by running multiple vLLM servers
across Ray workers. Each worker runs an independent vLLM server that can handle requests
in parallel.
"""
import asyncio
import gc
import os
import threading
from typing import List, Dict, Any
import multiprocessing as mp

from openai import OpenAI
import ray
import torch
from transformers import AutoTokenizer

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import FlexibleArgumentParser

from rllm.globals import BASE_SAMPLER_PORT
from rllm.sampler import SampleBatch
from rllm.sampler.utils import (
    convert_openai_response_to_samples,
    kill_process_and_children,
    wait_for_server,
)

OPTIMIZED_VLLM_CONFIG = {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 2**16,
    "enable_chunked_prefill": True,
    "num_scheduler_steps": 40,
    "enable_prefix_caching": True,
}

def _launch_vllm_server(args: Any) -> None:
    """Target function for our child process.
    
    This is where we call the blocking run_server(args).
    Because this runs in a separate process, it doesn't block the Ray actor's thread.
    """
    print(f"[Child Process] Launching vLLM server on port {args.port}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server(args))
    loop.close()

@ray.remote(num_gpus=None, num_cpus=None)
class RayVLLMWorker:
    """A Ray actor that runs a vLLM server.

    Each worker runs an independent vLLM server in a separate thread that can handle
    inference requests.

    Attributes:
        host: Host address to bind the server to
        port: Port to run the server on
        model_kwargs: Additional arguments passed to the vLLM model
    """

    def __init__(self, port: int, **model_kwargs) -> None:
        """Initialize the worker and prepare vLLM server configuration.

        Args:
            port: Port to run the server on
            **model_kwargs: Additional arguments for model configuration
        """
        self.port = port
        self.model = model_kwargs.get("model", "facebook/opt-125m")
        
        # Initialize server args
        openai_parser = make_arg_parser(FlexibleArgumentParser())
        self.server_args = openai_parser.parse_args([])
        
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

        # We'll store a reference to the child Process
        self.server_process = None
        self.oai_client = OpenAI(api_key='EMPTY', base_url=f"http://0.0.0.0:{self.port}/v1")
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
        
        # Wait for server to become healthy
        wait_for_server(self.port)
        return f"Started server on port {self.port}"
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> SampleBatch:
        try:
            chat_response = self.oai_client.chat.completions.create(
                model=self.model,
                messages=messages,
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
        
        prompt= self.tokenizer.apply_chat_template(messages, tokenize=False)
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
        """Terminate the vLLM server process (if running).

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

class DistributedVLLM:
    """Manages a distributed system of vLLM workers using Ray.

    This class handles:
    - Starting/stopping Ray cluster
    - Managing worker processes
    - Routing inference requests
    - Resource cleanup
    """

    WORKER_NAME_TEMPLATE = "persistent_vllm_worker_{}"
    NAMESPACE = "vllm_workers"

    def __init__(self, num_workers: int = 1, **model_kwargs) -> None:
        """Initialize the distributed system.

        Args:
            num_workers: Number of worker processes to launch
            **model_kwargs: Model configuration parameters
        """
        self._worker_lock = threading.Lock()
        self.model = model_kwargs.get("model", "facebook/opt-125m")
        self.num_workers = num_workers
        self.model_kwargs = model_kwargs
        self.active_requests = {}  # Dict to track active requests per worker

        try:
            ray.init(address="auto", namespace=self.NAMESPACE)
            print("Ray cluster is already initialized.")
        except ConnectionError:
            print("Starting new Ray cluster...")
            tensor_parallel_size = model_kwargs.get("tensor_parallel_size", 1)
            # Get total GPUs on cluster, programatically
            total_gpus = torch.cuda.device_count()
            total_cpus = os.cpu_count()
            os.system(f"ray start --head --num-cpus={total_cpus} --num-gpus={total_gpus}")
            ray.init(address="auto", namespace=self.NAMESPACE)

        # Fetch vLLM workers
        self.get_or_launch_workers()
        print("All vLLM servers have started successfully.")

    def get_or_launch_workers(self) -> None:
        """Launch vLLM workers or connect to existing ones."""
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
                worker = RayVLLMWorker.options(
                    name=worker_name,
                    lifetime="detached",
                    namespace=self.NAMESPACE,
                    num_gpus=self.model_kwargs.get("tensor_parallel_size", 1),
                    num_cpus=8,
                    max_concurrency=64,
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

    def _get_least_busy_worker(self) -> str:
        """Get the URL of the worker with the least number of active requests."""
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
            Dict containing the API response
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
    
    def restart(self):
        self.shutdown()
        self.get_or_launch_workers()

    def shutdown(self):
        """Shutdown the distributed system and clean up resources."""
        print("Shutting down vLLM workers...")
        for worker in self.workers:
            # Kill worker 
            ray.get(worker.shutdown.remote())
            ray.kill(worker)

if __name__ == "__main__":
    # Testing DistributedVLLM
    distributed_vllm = DistributedVLLM(
        num_workers=1,
        tensor_parallel_size=2,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    print(distributed_vllm.chat_completion([{"role": "user", "content": "What is the capital of France?"}]))
    print(distributed_vllm.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_vllm.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
    print(distributed_vllm.chat_completion([{"role": "user", "content": "What is the capital of MarioLand?"}]))
