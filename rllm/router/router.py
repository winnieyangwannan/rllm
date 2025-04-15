
import ray



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

    def _get_worker_idx(self, application_id):
        if application_id not in self.cache_map:
            self.cache_map[application_id] = self.next_placement
            self.next_placement = (self.next_placement + 1) % self.world_size
        return self.cache_map[application_id]

    async def _get_result_verl_async(self, batch, application_id, **kwargs):
        """
        Asynchronous version for getting a single action from verl using Ray worker groups.
        """
        # Execute the generation on a worker asynchronously
        obj_ref = self.rollout_engine.execute_worker_async(
            worker_idx=self._get_worker_idx(application_id),
            method_name='generate',
            prompts=batch
        )
        
        # Wait for the result
        done_refs, _ = ray.wait([obj_ref], num_returns=1)
        output = ray.get(done_refs[0])
        return output
       
    async def _get_result_verl_async_v2(self, batch, application_id, **kwargs):
        """
        Asynchronous version for getting a single action from verl using Ray worker groups.
        """
        # Execute the generation on a worker asynchronously

        if self.tp_size == 1:

            worker_idx = self._get_worker_idx(application_id)
            obj_ref = self.rollout_engine.execute_worker_async(
                worker_idx=worker_idx,
                method_name='generate_async',
                prompts=batch
            )
            
            # Wait for the result
            output = await obj_ref
            return output[0]

        
        # When tp > 1, schedule the request to all tp workers
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
        done_refs, _ = ray.wait(obj_refs, num_returns=self.tp_size)
        outputs = ray.get(done_refs)
        
        # Return the first result (all workers should return the same result when using tensor parallelism)
        return outputs[0][0]