from rllm.rllm.models.agent_execution_engine import AgentExecutionEngine
import asyncio
from verl.trainer.ppo.ray_trainer import _timer
from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward, compute_training_score, compute_environment_score

import torch
import numpy as np
import time
import os

import openai
import ray


class Scheduler:
    def __init__(
        self,
        rollout_engine,
    ):
        self.rollout_engine = rollout_engine

        self.world_size = self.rollout_engine.world_size
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
            worker_idx=self._get_worker_idx(application_id),  # Use the first worker
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
        obj_ref = self.rollout_engine.execute_worker_async(
            worker_idx=self._get_worker_idx(application_id),  # Use the first worker
            method_name='generate_async',
            prompts=batch
        )
        
        # Wait for the result
        done_refs, _ = ray.wait([obj_ref], num_returns=1)
        output = ray.get(done_refs[0])
        return output
       