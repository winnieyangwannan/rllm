from rllm.models.batch_agent import BatchAgent
import asyncio
from verl.trainer.ppo.ray_trainer import _timer
from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward, compute_training_score, compute_environment_score

import torch
import numpy as np
import time

import openai
from openai import AsyncOpenAI
client = AsyncOpenAI()


class AgentExecutionEngine(BatchAgent):
    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        agent_class,
        model_path="",
        n_parallel_agents=1,
        env=None,
        gamma=0.95,
        api_retries=3,
        retry_limit=1,
        episode_len=5,
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
        self.env = env
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.episode_len = episode_len
        self.agent_args = agent_args
        self.max_trajectory_length = max_trajectory_length

        self.agents = [agent_class(**agent_args) for _ in range(n_parallel_agents)]
        self.envs = [self.env for _ in range(self.n_parallel_agents)]

        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", None)

        if engine_name == "openai":
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
        Asynchronous version for getting a single action from verl.
        """
        from verl.protocol import pad_dataproto_to_divisor
        import asyncio

        prompt = agent._pre_get_action(trajectory)
        batch = self._convert_prompt_verl([prompt], **kwargs)
        
        # Pad to match worker group world size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.rollout_engine.world_size)
        
        # Add rollout_generator_id (-1 for padding)
        rollout_generator_id = np.array([0] + [-1] * pad_size, dtype=object)
        batch_padded.non_tensor_batch["rollout_generator_id"] = rollout_generator_id
        
        # Run the generator in a separate thread
        result = None
        
        def collect_result():
            nonlocal result
            gen_seq_generator = self.rollout_engine.generate_sequences_async(prompts=batch)
            for output in gen_seq_generator:
                idx = output.non_tensor_batch["rollout_generator_id"][0]
                if idx != -1:  # Skip padding
                    output_text = self.tokenizer.batch_decode(
                        output.batch["responses"], skip_special_tokens=False
                    )[0]  # Only one response
                    pad_token = self.tokenizer.pad_token
                    response = output_text.replace(pad_token, "")
                    result = agent._post_get_action(response)
                    break  # We only need the first result
        
        # Run collection in a separate thread
        await asyncio.to_thread(collect_result)
        
        return result

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


    async def execute_tasks(self, tasks):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.
        
        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)
            
        Returns:
            A list of trajectories, one for each task.
        """

        max_concurrent = self.n_parallel_agents
        
        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}
        
        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        active_requests = []
        
        async def run_agent_episode(task_id, task, agent_idx, env_idx):
            """Run a single agent's episode asynchronously"""
            agent = self.agents[agent_idx]
            env = self.envs[env_idx]
            
            # Reset environment with the task
            observation, _ = env.reset(task=task)
            
            # Reset agent
            agent.reset()
            
            # Initialize trajectory for this task
            trajectory = []
            
            for _ in range(self.episode_len):
                trajectory.append({
                    "next_observation": observation,
                })

                # Get action from agent
                action = await self.get_action_async(trajectory, agent)
                
                # Take step in environment
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Update agent
                agent.update(
                    action=action,
                    observation=observation,
                    next_observation=next_observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info
                )
                
                observation = next_observation
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            return task_id, trajectory
        
        # Initialize the first batch of tasks
        for i in range(min(max_concurrent, len(task_queue))):
            task_id, task = task_queue.pop(0)
            agent_idx = i % len(self.agents)
            env_idx = i % len(self.envs)
            
            task_coroutine = run_agent_episode(task_id, task, agent_idx, env_idx)
            active_requests.append((asyncio.create_task(task_coroutine), agent_idx, env_idx))
        
        # Process tasks and refill the active requests as they complete
        while active_requests:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                [task for task, _, _ in active_requests],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            completed_info = None
            for task, agent_idx, env_idx in active_requests:
                if task in done:
                    completed_info = (agent_idx, env_idx)
                    break
                    
            # Update active_requests with pending tasks
            active_requests = [(task, agent_idx, env_idx) for task, agent_idx, env_idx in active_requests 
                              if task in pending]
            
            # Process completed tasks and add new ones
            for completed_task in done:
                task_id, trajectory = await completed_task
                all_trajectories[task_id] = trajectory
                
                # If there are more tasks in the queue, add a new one
                if task_queue and completed_info is not None:
                    new_task_id, new_task = task_queue.pop(0)
                    available_agent, available_env = completed_info
                    
                    # Create a new task with the available agent and environment
                    new_coroutine = run_agent_episode(new_task_id, new_task, available_agent, available_env)
                    active_requests.append((asyncio.create_task(new_coroutine), available_agent, available_env))
        
        # Convert the dictionary to a list ordered by task_id
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories