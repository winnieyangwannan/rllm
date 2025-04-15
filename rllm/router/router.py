
import ray
import asyncio


class Router:
    def __init__(
        self,
        rollout_engine,
    ):
        self.rollout_engine = rollout_engine

        self.world_size = self.rollout_engine.world_size
        self.tp_size = getattr(self.rollout_engine, 'tp_size', 1)

        self.next_placement = 0
        self.cache_map = {} # application id to underlying engine mapping, for verl it's id->worker

        # used to synchronize verl sharding manager enter and exit
        self.lock = asyncio.Lock()
        self.active_requests = 0

    def _get_worker_idx(self, application_id):
        if application_id not in self.cache_map:
            self.cache_map[application_id] = self.next_placement
            self.next_placement = (self.next_placement + self.tp_size) % self.world_size
        return self.cache_map[application_id]

    # The two functions below are to invoke sharding managers concurrently so all_gather doesn't hang
    async def __enter__(self):
        async with self.lock:
            if self.active_requests == 0:
                self.rollout_engine.generate_async_sharding_manager_enter()
            self.active_requests += 1

    async def __exit__(self):
        async with self.lock:
            self.active_requests -= 1
            if self.active_requests == 0:
                self.rollout_engine.generate_async_sharding_manager_exit()
       
    async def _get_result_verl_async(self, batch, application_id, **kwargs):
        """
        Asynchronous version for getting a single action from verl using Ray worker groups.
        """
        

        # # Execute the generation on a worker asynchronously
        # if self.tp_size == 1:
        #     worker_idx = self._get_worker_idx(application_id)
        #     obj_ref = self.rollout_engine.execute_worker_async(
        #         worker_idx=worker_idx,
        #         method_name='generate_async',
        #         prompts=batch,
        #         **kwargs
        #     )
        #     # Wait for the result
        #     output = await asyncio.to_thread(ray.get, obj_ref)
        #     return output[0]
        
        # Get the base worker index for this application
        base_worker_idx = self._get_worker_idx(application_id)
            
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
        done_refs, _ = await asyncio.to_thread(ray.wait, obj_refs, num_returns=self.tp_size)
        outputs = await asyncio.to_thread(ray.get, done_refs)
        # Return the first result (all workers should return the same result when using tensor parallelism)
        return outputs[0][0]