import ray
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Router:
    def __init__(
        self,
        rollout_engine,
        tensor_parallel_size=1,
    ):
        self.rollout_engine = rollout_engine

        self.world_size = self.rollout_engine.world_size
        self.tp_size = tensor_parallel_size

        self.next_placement = 0
        self.cache_map = {} # application id to underlying engine mapping, for verl it's id->worker
        self._lock = asyncio.Lock()
        # Initialize a dedicated ThreadPoolExecutor for ray.get calls
        self._ray_get_executor = None
        # Maps number of active requests for each replica/engine.
        self.request_dict = {i: 0 for i in range(self.world_size//self.tp_size)}


    async def _get_worker_idx(self, application_id):
        """
        Get the worker index for a given application ID in a thread-safe manner.
        Uses "Autellix/Agentix" load balancing: Least Used + Data locality policy w.r.t Program ID.
        """
        async with self._lock:
            # Get engine idx with least number of requests
            min_idx, min_requests = min(self.request_dict.items(), key=lambda x: x[1])
            if application_id not in self.cache_map:
                self.cache_map[application_id] = min_idx
            elif self.request_dict[self.cache_map[application_id]] - min_requests >= 4: # Put on least used engine if skew is too large
                self.cache_map[application_id] = min_idx
            engine_idx = self.cache_map[application_id]
            self.request_dict[engine_idx] += 1
            return engine_idx

    def __enter__(self):
        self.rollout_engine.generate_async_sharding_manager_enter()
        # Create ray.get executor
        self._ray_get_executor = ThreadPoolExecutor(
            max_workers=1024,
            thread_name_prefix='RayGetExecutor'
        )

    def __exit__(self): # Corrected signature
        self.rollout_engine.generate_async_sharding_manager_exit()
        # Kill ray.get executor
        self._ray_get_executor.shutdown(wait=False)  # Fixed typo

    async def _get_result_verl_async(self, batch, application_id, **kwargs):
        """
        Asynchronous version for getting a LLM response from verl using Ray worker groups.
        """
        # Get the base engine index for this application using the thread-safe method
        base_engine_idx = await self._get_worker_idx(application_id)
        # Create a list of engine indices to use (wrapping around if needed)
        engine_indices = [(base_engine_idx + i) % self.world_size for i in range(self.tp_size)]

        print(f"Request dict: {self.request_dict}")

        # Execute the generation on multiple workers asynchronously
        obj_refs = [
            self.rollout_engine.execute_worker_async(
                worker_idx=idx,
                method_name="generate_async",
                prompts=batch,
                **kwargs
            )
            for idx in engine_indices
        ]
        # Wait for all results
        loop = asyncio.get_running_loop() # Get current event loop
        outputs = await loop.run_in_executor(self._ray_get_executor, ray.get, obj_refs)

        async with self._lock:
            self.request_dict[base_engine_idx] -= 1

        return outputs[0][0]