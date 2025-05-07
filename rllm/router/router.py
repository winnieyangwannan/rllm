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
        self._ray_get_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='RayGetExecutor'
        )


    async def _get_worker_idx(self, application_id):
        """
        Get the worker index for a given application ID in a thread-safe manner.
        Uses "Autellix" load balancing: Round Robin + Data locality policy w.r.t Program ID.
        """
        if application_id in self.cache_map:
             return self.cache_map[application_id]

        # Acquire the lock to ensure exclusive access to cache_map and next_placement
        async with self._lock:
            if application_id not in self.cache_map:
                self.cache_map[application_id] = self.next_placement
                self.next_placement = (self.next_placement + self.tp_size) % self.world_size
            return self.cache_map[application_id]

    def __enter__(self):
        self.rollout_engine.generate_async_sharding_manager_enter()

    def __exit__(self): # Corrected signature
        self.rollout_engine.generate_async_sharding_manager_exit()
        # Shutdown the dedicated executor
        self._ray_get_executor.shutdown(wait=True)

    async def _get_result_verl_async(self, batch, application_id, **kwargs):
        """
        Asynchronous version for getting a LLM response from verl using Ray worker groups.
        """
        # Get the base worker index for this application using the thread-safe method
        base_worker_idx = await self._get_worker_idx(application_id)
        # Create a list of worker indices to use (wrapping around if needed)
        worker_indices = [(base_worker_idx + i) % self.world_size for i in range(self.tp_size)]

        # Execute the generation on multiple workers asynchronously
        obj_refs = [
            self.rollout_engine.execute_worker_async(
                worker_idx=idx,
                method_name="generate_async",
                prompts=batch,
                **kwargs
            )
            for idx in worker_indices
        ]
        # Wait for all results
        loop = asyncio.get_running_loop() # Get current event loop
        outputs = await loop.run_in_executor(self._ray_get_executor, ray.get, obj_refs)
        return outputs[0][0]