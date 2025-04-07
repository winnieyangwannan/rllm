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


class AsyncAgentExecutionEngine(AgentExecutionEngine):
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

    async def run_agent_episode(self, task_id, task, agent_idx, env_idx):
        """Run a single agent's episode asynchronously"""
        agent = self.agents[agent_idx]
        env = self.envs[env_idx]
        
        # Reset environment with the task
        observation, _ = env.reset(task=task)
        
        # Reset agent
        agent.reset()
        
        # Initialize trajectory for this task
        trajectory = []
        
        for _ in range(self.max_episode_len):
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
    
    async def run_agent_episode_verl(self, task_id, task, agent_idx, env_idx):
        """Run a single agent's episode asynchronously"""
        agent = self.agents[agent_idx]
        env = self.envs[env_idx]
        
        # Reset environment with the task
        observation, _ = env.reset(task=task)
        
        # Reset agent
        agent.reset()
        
        # Initialize trajectory for this task
        trajectory = []
        
        # For tracking token lengths
        prompt_tokens = []
        response_tokens = []
        response_masks = []
        
        # Initialize with first observation
        trajectory.append({
            "next_observation": observation,
        })
        
        # Format initial observation as messages
        initial_messages = agent.format_observation_as_messages(observation)
        prompt_tokens, _ = self._convert_messages_to_tokens_and_masks(initial_messages)
        
        # Track all messages for conversation history
        all_messages = initial_messages.copy()
        
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < self.max_episode_len:
            steps += 1
            
            # Get action from agent
            action, response = await self.get_action_async(trajectory, agent)
            
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
            
            # Process response tokens
            assistant_msg = {"role": "assistant", "content": response}
            env_messages = agent.format_observation_as_messages(next_observation)

            print("assistant_msg:", assistant_msg)
            print("env_messages:", env_messages)
            
            assistant_msg_tokens, assistant_msg_masks = self._convert_message_to_tokens_and_masks(assistant_msg)
            env_msg_tokens, env_msg_masks = self._convert_messages_to_tokens_and_masks(env_messages)
            
            # Update token collections
            response_tokens.extend(assistant_msg_tokens + env_msg_tokens)
            response_masks.extend(assistant_msg_masks + env_msg_masks)
            
            # Update conversation history
            all_messages.append(assistant_msg)
            all_messages.extend(env_messages)
            
            # Update trajectory
            trajectory.append({
                "observation": observation,
                "next_observation": next_observation,
                "reward": reward,
                "done": terminated or truncated,
                "action": action,
                "info": info,
                "response": action,
                "truncated": False,
            })
            
            observation = next_observation
        
        # Process trajectory for training
        processed_trajectory = trajectory[1:]  # Remove sentinel
        augmented_trajectory = add_mc_return(add_trajectory_reward(processed_trajectory), gamma=self.gamma)
        training_reward = agent.compute_training_reward(augmented_trajectory)
        final_trajectory = add_training_reward(augmented_trajectory, training_reward)
        
        # Create token result similar to interact_environment
        token_result = {
            "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
            "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
            "response_masks": torch.tensor(response_masks, dtype=torch.long),
            "training_reward": compute_training_score(final_trajectory),
            "environment_reward": compute_environment_score(final_trajectory),
        }
        
        return task_id, final_trajectory, token_result, all_messages

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
        
        # Initialize the first batch of tasks
        for i in range(min(max_concurrent, len(task_queue))):
            task_id, task = task_queue.pop(0)
            agent_idx = i % len(self.agents)
            env_idx = i % len(self.envs)
            
            task_coroutine = self.run_agent_episode(task_id, task, agent_idx, env_idx)
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
                    new_coroutine = self.run_agent_episode(new_task_id, new_task, available_agent, available_env)
                    active_requests.append((asyncio.create_task(new_coroutine), available_agent, available_env))
        
        # Convert the dictionary to a list ordered by task_id
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories