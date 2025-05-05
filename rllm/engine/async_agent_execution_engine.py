import asyncio
import uuid
import concurrent.futures

import openai
import torch

from rllm.agents.agent import Step, Trajectory
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.env_utils import (
    compute_trajectory_reward,
    compute_mc_return,
)

from rllm.misc import colorful_print
from rllm.parser.chat_template.parser import ChatTemplateParser
from rllm.router.router import Router
from rllm.agents.utils import get_recent_assistant_user_messages, convert_messages_to_tokens_and_masks
from rllm.router.router import Router
import torch
import uuid
from typing import List, Dict


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        config,
        agents=[],
        envs=[],
        model_path="",
        n_parallel_agents=1,
        gamma=1.0,
        api_retries=3,
        retry_limit=1,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        agent_class=None,
        env_class=None,
        agent_args={},
        rollout_engine_args={},
        env_args={},
        **kwargs,
    ):
        self.config = config
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.model_path = model_path

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.agent_args = agent_args
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.agents = agents
        self.env_class = env_class
        self.env_args = env_args
        self.envs = envs

        # if agent_class is not None and env_class is not None:
        #     self.agents = [agent_class(**agent_args) for _ in range(n_parallel_agents)]
        #     self.envs = [env_class(**env_args) for _ in range(n_parallel_agents)]
        # else:
        #     self.agents = agents
        #     self.envs = envs

        assert all(type(env).is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"
        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", None)

        if self.engine_name == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(**self.rollout_engine_args) 
        elif self.engine_name == "verl":
            # All generation is done via scheduler. Currently only works for verl
            self.router = Router(rollout_engine=rollout_engine, tensor_parallel_size=self.config.actor_rollout_ref.rollout.get('tensor_model_parallel_size', 1))

        self.chat_template_parser = ChatTemplateParser.get_parser(self.tokenizer)

    async def get_model_response(self, prompt, application_id, **kwargs):
        """
        Compute model response asynchronously based on the engine type.
        
        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.
        
        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The model's response text
            
        Raises:
            NotImplementedError: If the engine type is not supported
        """
        if self.engine_name == "openai":
            return await self._get_openai_async(prompt, application_id, **kwargs)
        elif self.engine_name == "verl":
            return await self._get_verl_async(prompt, application_id, **kwargs)
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    async def _get_verl_async(self, prompt, application_id, **kwargs):
        """
        Get action from VERL asynchronously using Ray worker groups.
        
        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The processed response text with padding tokens removed
        """
        batch = self._convert_prompt_verl([prompt], **kwargs)
        
        if 'max_tokens' in kwargs:
            batch.meta_info['max_tokens'] = kwargs['max_tokens']

        output = await self.router._get_result_verl_async(
            batch, application_id, **kwargs
        )

        attn = output.batch["attention_mask"][0, self.max_prompt_length:]
        tokens = output.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[:last_valid_idx + 1]  # include the last valid token

        response = self.tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = self.tokenizer.pad_token
        eos_token = self.tokenizer.eos_token
        response = response.replace(pad_token, "").replace(eos_token, "")
        return response

    async def _get_openai_async(self, prompt, _, **kwargs):
        """
        Get action from OpenAI API asynchronously with retry logic.
        
        Args:
            prompt: The input prompt in chat completions format
            application_id: Unique identifier for the application (unused for OpenAI)
            **kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            The response from OpenAI API
        """
        async def get_response(prompt: List[Dict[str, str]]):
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
                    print("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    print("Error: ", e)
                    return f"Error processing content: {e}"

        response = await get_response(prompt)
        return response

    async def run_agent_trajectory(
        self, idx, application_id, seed=0, mode="Text", **kwargs
    ):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        loop = asyncio.get_running_loop()  # Get the current event loop

        # Initialize trajectory for this task.
        trajectory = []
        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []

        # Reset environment with the task using the executor
        observation, info = await asyncio.to_thread(env.reset)
        info['max_steps'] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation, # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages,
                                                                tokenizer=self.tokenizer,
                                                                parser=self.chat_template_parser,
                                                                contains_first_msg=True,
                                                                contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length already exceeded max_prompt_length, retrying"
            )

        for step_idx in range(self.max_steps):
            print(f"Trajectory {idx}, Step {step_idx}/{self.max_steps}")
            # Get action from agent
            chat_completions_messages = agent.chat_completions
            # Max remaining tokens left for the response
            kwargs['max_tokens'] = self.max_response_length - response_token_len
            response = await self.get_model_response(
                chat_completions_messages,
                application_id,
                **kwargs
            )
            agent.update_from_model(response)
            # Fetch action from agent's internal state.
            cur_state = agent.get_current_state()
            action = cur_state.action
            # Take step in environment using the executor
            next_observation, reward, done, info = await asyncio.to_thread(env.step, action)
            info['max_steps'] = self.max_steps
            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message],
                                                                                            tokenizer= self.tokenizer,
                                                                                            parser=self.chat_template_parser,
                                                                                            contains_first_msg=False,
                                                                                            contains_generation_msg=False)
            env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages,
                                                                                tokenizer=self.tokenizer,
                                                                                parser=self.chat_template_parser,
                                                                                contains_first_msg=False,
                                                                                contains_generation_msg=True)
            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[
                    :truncation_length
                ]
                truncated_response_masks = (assistant_msg_masks + env_msg_masks)[
                    :truncation_length
                ]
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)
                
                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)
            
            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )

        # Closing environment using the executor.
        await asyncio.to_thread(env.close)
        trajectory = agent.trajectory
        
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "training_reward": agent.compute_training_reward(trajectory) if hasattr(agent, "compute_training_reward") else trajectory.steps[-1].reward,
                "environment_reward": trajectory.reward,
                "idx": env.idx,
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions

    async def run_agent_trajectory_with_retry(
        self, idx, application_id, seed=0, mode="Text", **kwargs
    ):
        for _ in range(self.retry_limit):
            try:
                return await self.run_agent_trajectory(
                    idx, application_id=application_id, seed=seed, mode=mode, **kwargs
                )
            except Exception as e:
                print(e)
                continue
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(
        self, reset_seed=0, timing_raw={}, mode="Text", **kwargs
    ):
        assert all(type(env).is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"
        # Note: this function is not concurrecy safe due to the router.__enter__ and router.__exit__
        if self.engine_name == "verl":
            self.router.__enter__()

        application_ids = [str(uuid.uuid4()) for _ in range(self.n_parallel_agents)]

        tasks = [
            self.run_agent_trajectory_with_retry(
                i,
                application_id=application_ids[i],
                seed=reset_seed,
                mode=mode,
                **kwargs,
            )
            for i in range(self.n_parallel_agents)
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                raise e
        
        if self.engine_name == "verl":
            self.router.__exit__()

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
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        async def sem_wrapper(task_id, task):
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index].reset(task=task)
                    res = await self.run_agent_trajectory(index, task_id)
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)
        
        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])
        
        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories