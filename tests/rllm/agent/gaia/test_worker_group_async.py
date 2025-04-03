# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
e2e test verl.single_controller.ray
"""

import torch
import ray

from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.decorator import register, Dispatch, collect_all_to_all, Execute
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import AsyncLLMEngine, LLM, SamplingParams, TokensPrompt

import uuid

def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x
        self.inference_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model="Qwen/Qwen2.5-Math-1.5B", # Use a common test model
                    enable_sleep_mode=True,
                    tensor_parallel_size=1, # Single GPU for testing
                    dtype="float16", # Use fp16 for testing
                    enforce_eager=True,
                    gpu_memory_utilization=0.8,
                    disable_custom_all_reduce=True,
                    skip_tokenizer_init=False,
                    max_model_len=2048, # Reasonable context length for testing
                    disable_log_stats=True,
                    max_num_batched_tokens=4096,
                    enable_chunked_prefill=True,
                    enable_prefix_caching=True
                )
            )

    async def foo_async(self, y):
        import asyncio
        await asyncio.sleep(5)
        return self._x + y
    
    async def generate(self, prompt: str = "Hello, how are you?", sampling_params: SamplingParams = None):
        """Generate text using the inference engine.
        
        Args:
            prompt: The input prompt to generate from
            sampling_params: Optional sampling parameters, uses defaults if not provided
        
        Returns:
            The generated text output
        """
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
            
        request_id = str(uuid.uuid4())
        
        # The generate method returns an async generator, so we need to iterate over it
        outputs = None
        async for output in self.inference_engine.generate(prompt, 
                                                         sampling_params,
                                                         request_id):
            outputs = output  # Keep the last output
        
        if outputs is None:
            return "No output generated"
            
        # generated_text = self.inference_engine.tokenizer.decode(outputs.token_ids)
        print(f"Generated text: {outputs}")
        return outputs


    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        return self._x + y + x

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def foo_one_to_all(self, x, y):
        return self._x + y + x

    @register(Dispatch.ALL_TO_ALL, blocking=False)
    def foo_all_to_all(self, x, y):
        return self._x + y + x

    @register(dispatch_mode={'dispatch_fn': two_to_all_dispatch_fn, 'collect_fn': collect_all_to_all})
    def foo_custom(self, x, y):
        return self._x + y + x


@ray.remote(num_gpus=0.1)
def remote_call_wg(worker_names):
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
    worker_group = RayWorkerGroup.from_detached(worker_names=worker_names, ray_cls_with_init=class_with_args)
    print(worker_group.worker_names)

    output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
    assert output_ref == [8, 10, 8, 10]

    output_ref = worker_group.foo_rank_zero(x=1, y=2)
    assert output_ref == 5

    return worker_group.worker_names


def add_one(data):
    data = data.to("cuda")
    data += 1
    data = data.to("cpu")
    return data


def test_basics():
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([2], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)

    worker_group = RayWorkerGroup(resource_pool=resource_pool,
                                  ray_cls_with_init=class_with_args,
                                  name_prefix="worker_group_basic")

    print(worker_group.worker_names)

    # output = worker_group.execute_all_async("foo", y=3)
    # print(output)
    
    # Test async request scheduling
    def test_request_scheduler():
        # Sample batch of requests
        requests = [{"prompt": "Hello, how are you?"} for i in range(10)]
        queue_size = 4  # Match number of workers
        
        # Track completed requests and their results
        completed_results = []
        pending_refs = {}  # Maps object_ref to request index
        
        # Initial filling of the queue
        queue_count = 0
        for i in range(min(queue_size, len(requests))):
            worker_idx = i % worker_group.world_size
            obj_ref = worker_group.execute_worker_async(worker_idx, "generate", **requests[i])
            pending_refs[obj_ref] = i
            queue_count += 1
        
        # Process results as they complete and add new tasks
        while pending_refs:
            # Wait for any task to complete
            done_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
            done_ref = done_refs[0]
            
            # Get result and track it
            result = ray.get(done_ref)
            request_idx = pending_refs.pop(done_ref)
            completed_results.append((request_idx, result))
            print(f"Request {request_idx} completed with result: {result}")
            
            # Add a new task if there are more requests
            next_idx = queue_count
            if next_idx < len(requests):
                worker_idx = next_idx % worker_group.world_size
                obj_ref = worker_group.execute_worker_async(worker_idx, "generate", **requests[next_idx])
                pending_refs[obj_ref] = next_idx
                queue_count += 1
        
        # Verify all requests were processed
        assert len(completed_results) == len(requests)
        # Results might not be in order due to async execution
        completed_results.sort(key=lambda x: x[0])
        print("All requests processed successfully!")
    
    # Run the scheduler test
    test_request_scheduler()

    ray.shutdown()


if __name__ == '__main__':
    test_basics()
