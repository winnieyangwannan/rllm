import gc
from typing import List

import ray
import torch
from vllm import LLM, SamplingParams

# @ray.remote(num_gpus=None)
# class RayVLLMWorker:
#     def __init__(self, tensor_parallel_size: int = 1, **model_kwargs):
#         model_kwargs['tensor_parallel_size'] = tensor_parallel_size
#         self.model = LLM(**model_kwargs)

#     def chat(self, messages, sampling_params):
#         return self.model.chat(messages, sampling_params)

#     def shutdown(self):
#         del self.model
#         gc.collect()
#         torch.cuda.empty_cache()

@ray.remote(num_gpus=None)
class RayVLLMWorker:
    def __init__(self, tensor_parallel_size: int = 1, **model_kwargs):
        model_kwargs['tensor_parallel_size'] = tensor_parallel_size
        self.model = LLM(**model_kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.model_kwargs = model_kwargs

    def get_config(self):
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "model_kwargs": self.model_kwargs
        }

    def chat(self, messages, sampling_params):
        return self.model.chat(messages, sampling_params)

    def shutdown(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class DistributedVLLM:
    WORKER_NAME_TEMPLATE = "persistent_vllm_worker_{}"
    NAMESPACE = "vllm_workers"

    def __init__(self, num_workers: int, tensor_parallel_size: int = 1, **model_kwargs):
        try:
            ray.init(address="auto")
        except:
            print("ray is not initialized")
            total_gpus = num_workers * tensor_parallel_size
            ray.init(num_gpus=total_gpus, namespace=self.NAMESPACE)


        self.workers = []
        for i in range(num_workers):
            worker_name = self.WORKER_NAME_TEMPLATE.format(i)
            try:
                # Try to get existing worker
                worker = ray.get_actor(worker_name, namespace=self.NAMESPACE)

                # Verify configuration matches
                config = ray.get(worker.get_config.remote())
                # if (config["tensor_parallel_size"] != tensor_parallel_size or 
                #     config["model_kwargs"] != model_kwargs):
                #     print("# Configuration mismatch - shutdown and recreate")
                #     ray.get(worker.shutdown.remote())
                #     worker = self._create_worker(worker_name, tensor_parallel_size, model_kwargs)
            except ValueError as e:

                print(e)
                # Worker doesn't exist, create new one
                worker = self._create_worker(worker_name, tensor_parallel_size, model_kwargs)

                print("# Worker doesn't exist, create new one")
            
            self.workers.append(worker)

    def _create_worker(self, name, tensor_parallel_size, model_kwargs):
        return RayVLLMWorker.options(
            name=name,
            lifetime="detached",  # This makes the actor persist
            num_gpus=tensor_parallel_size
        ).remote(
            tensor_parallel_size=tensor_parallel_size,
            **model_kwargs
        )

    def shutdown(self, persist: bool = True):
        """
        Shutdown the distributed system
        Args:
            persist: If False, will shutdown all workers. If True, keeps workers alive.
        """
        if not persist:
            for worker in self.workers:
                ray.get(worker.shutdown.remote())
        ray.shutdown()


    def chat(self, messages: List, sampling_params: SamplingParams):
        # Split the messages into chunks for each worker
        chunk_size = (len(messages) + len(self.workers) - 1) // len(self.workers)
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        
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