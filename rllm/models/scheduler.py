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
        engine_name,
        tokenizer,
        agent_class,
        env_class,
        model_path="",
        n_parallel_agents=1,
        gamma=0.95,
        api_retries=3,
        retry_limit=1,
        max_episode_len=5,
        max_trajectory_length=8000,
        agent_args={},
        rollout_engine_args={},
        env_args={},
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.agent_class = agent_class
        self.n_parallel_agents = n_parallel_agents
        self.model_path = model_path

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_episode_len = max_episode_len
        self.agent_args = agent_args
        self.max_trajectory_length = max_trajectory_length

        self.agents = [agent_class(**agent_args) for _ in range(self.n_parallel_agents)]
        self.envs = [env_class(**env_args) for _ in range(self.n_parallel_agents)]

        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", None)

        if engine_name == "openai":
            from openai import AsyncOpenAI 
            self.client = AsyncOpenAI(**self.rollout_engine_args)


    async def get_action_async(self, trajectory, agent, **kwargs):
        if self.engine_name == "openai":
            return await self._get_action_openai_async(trajectory, agent, **kwargs)
        elif self.engine_name == "verl":
            return await self._get_action_verl_async(trajectory, agent, **kwargs)
        else:
            raise NotImplementedError


    async def _get_action_verl_async(self, trajectory, agent, **kwargs):
        """
        Asynchronous version for getting a single action from verl using Ray worker groups.
        """
        prompt = agent._pre_get_action(trajectory)
        batch = self._convert_prompt_verl([prompt], **kwargs)
        
        # Execute the generation on a worker asynchronously
        obj_ref = self.rollout_engine.execute_worker_async(
            worker_idx=0,  # Use the first worker
            method_name='generate',
            prompts=batch
        )
        
        # Wait for the result
        done_refs, _ = ray.wait([obj_ref], num_returns=1)
        output = ray.get(done_refs[0])
        
        # Process the output
        output_text = self.tokenizer.batch_decode(
            output.batch["responses"], skip_special_tokens=False
        )[0]  # Only one response
        
        pad_token = self.tokenizer.pad_token
        response = output_text.replace(pad_token, "")
        action = agent._post_get_action(response)
        
        return action, response

    async def _get_action_openai_async(self, trajectory, agent, **kwargs):
        prompt = agent._pre_get_action(trajectory)
        
        async def get_response(prompt):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.chat.completions.create(
                        messages=prompt,
                        **self.sampling_params,
                        **kwargs,
                    )

                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    print(f"Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    return f"Error processing content: {e}"

        response = await get_response(prompt)
        print("oai response:", response)
        return agent._post_get_action(response)    
