import gc
from typing import List

import ray
import torch
from vllm import LLM, SamplingParams

@ray.remote(num_gpus=None)
class RayVLLMWorker:
    def __init__(self, tensor_parallel_size: int = 1, **model_kwargs):
        model_kwargs['tensor_parallel_size'] = tensor_parallel_size
        self.model = LLM(**model_kwargs)

    def chat(self, messages, sampling_params):
        return self.model.chat(messages, sampling_params)

    def shutdown(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class DistributedVLLM:
    def __init__(self, num_workers: int, tensor_parallel_size: int = 1, **model_kwargs):
        total_gpus = num_workers * tensor_parallel_size
        ray.init(num_gpus=total_gpus)
        
        self.workers = [
            RayVLLMWorker.options(num_gpus=tensor_parallel_size).remote(
                tensor_parallel_size=tensor_parallel_size,
                **model_kwargs
            ) for _ in range(num_workers)
        ]

    def chat(self, messages: List, sampling_params: SamplingParams):
        # Split the messages into chunks for each worker
        chunk_size = (len(messages) + len(self.workers) - 1) // len(self.workers)
        chunks = [
            messages[i : i + chunk_size] for i in range(0, len(messages), chunk_size)
        ]

        # Dispatch chunks to workers
        futures = [
            worker.chat.remote(chunk, sampling_params)
            for worker, chunk in zip(self.workers, chunks)
        ]

        # Gather results
        results = ray.get(futures)

        # Combine results
        combined_results = []
        for result in results:
            combined_results.extend(result)

        return combined_results

    def shutdown(self):
        for worker in self.workers:
            ray.get(worker.shutdown.remote())
        ray.shutdown()
