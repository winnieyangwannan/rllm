import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from verl.trainer.ppo.ray_trainer import _timer

from rllm.misc import colorful_print
from rllm.parser.chat_template.parser import ChatTemplateParser

from rllm.environments.env_utils import (
    compute_trajectory_reward,
    compute_mc_return,
)
from rllm.agents.utils import get_recent_assistant_user_messages, convert_messages_to_tokens_and_masks


class AgentExecutionEngine:
    """Engine for executing agent interactions with environments."""

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        agents=None,  # List of agents
        envs=None,  # List of environments
        model_path="",
        gamma=0.95,
        api_retries=3,
        retry_limit=1,
        max_steps=5,
        max_prompt_length=2048,  # Max prompt length for agent is only applied to first request
        max_response_length=16384,
        rollout_engine_args=None,
        max_workers=16,
        enforce_max_prompt_length=False, # If enabled, applies max_prompt check per step
        **kwargs,
    ):
        """Initialize the agent execution engine.
        
        Args:
            rollout_engine: Engine for rolling out trajectories
            engine_name: Name of the engine to use
            tokenizer: Tokenizer for the model
            agents: List of agents
            envs: List of environments
            model_path: Path to the model
            gamma: Discount factor
            api_retries: Number of API retries
            retry_limit: Number of retry limits
            max_steps: Maximum number of steps
            max_prompt_length: Maximum prompt length
            max_response_length: Maximum response length
            rollout_engine_args: Arguments for the rollout engine
            **kwargs: Additional arguments
        """
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.model_path = model_path

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.max_workers = max_workers
        agents = agents or []
        envs = envs or []
        self.n_parallel_agents = len(envs)
        
        assert len(agents) == len(envs), (
            f"Number of agents must equal to number of environments but received, "
            f"{len(agents)} and {len(envs)}"
        )
        self.agents = agents
        self.envs = envs

        # rollout engine args
        self.rollout_engine_args = rollout_engine_args or {}
        self.sampling_params = kwargs.get("sampling_params", None)

        if engine_name == "openai":
            from openai import OpenAI 
            self.client = OpenAI(**self.rollout_engine_args)

        self.chat_template_parser = ChatTemplateParser.get_parser(self.tokenizer)
    
    def get_model_response(self, prompts, seq_idxs, **kwargs):
        """
        Compute model response based on the engine type.
        
        This function routes the request to the appropriate engine-specific handler.
        
        Args:
            prompts: List of input prompts to send to the model
            seq_idxs: List of indices indicating which agent each prompt is for
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            List of model response texts
            
        Raises:
            NotImplementedError: If the engine type is not supported
        """
        assert len(prompts) == len(seq_idxs), (
            f"Number of prompts {len(prompts)} should equal to the number of agents "
            f"they are for ({len(seq_idxs)})"
        )
        if self.engine_name == "verl":
            return self._get_verl_sync(prompts, seq_idxs, **kwargs)
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def _get_verl_sync(self, prompts, seq_idxs, **kwargs):
        """Get responses from veRL engine synchronously.
        
        Args:
            prompts: List of prompts to send to the model
            seq_idxs: List of indices indicating which agent each prompt is for
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Tuple of (responses, seq_idxs)
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        batch = self._convert_prompt_verl(prompts, **kwargs)

        # because of veRL's chunking. we need to pad number of prompts to be a multiple of worker group world size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.rollout_engine.world_size)
        if 'max_tokens' in kwargs:
            batch_padded.meta_info['max_tokens'] = kwargs['max_tokens']
        output_padded = self.rollout_engine.generate_sequences(batch_padded)
        
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        attention_mask = output.batch["attention_mask"][:, self.max_prompt_length:]
        responses_tokens = output.batch["responses"]

        responses = []
        batch_size = responses_tokens.size(0)

        for i in range(batch_size):
            tokens = responses_tokens[i]
            attn = attention_mask[i]

            # Find last index where attention == 1
            non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
            if len(non_pad_indices) == 0:
                trimmed = tokens[:0]  # empty
            else:
                last_valid_idx = non_pad_indices[-1].item()
                trimmed = tokens[:last_valid_idx + 1]  # include the last valid token

            text = self.tokenizer.decode(trimmed, skip_special_tokens=False)
            pad_token = self.tokenizer.pad_token
            eos_token = self.tokenizer.eos_token
            text = text.replace(pad_token, "").replace(eos_token, "")
            responses.append(text)
        return responses, seq_idxs

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL
        
        Args:
            prompts: List of prompts to convert
            **kwargs: Additional arguments
            
        Returns:
            DataProto object containing the converted prompts
        """
        from verl.utils.model import compute_position_id_with_mask
        from verl import DataProto
        from verl.protocol import union_two_dict
        from verl.utils.torch_functional import pad_sequence_to_length
        
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        formatted_prompts = [
            self.chat_template_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) 
            for prompt in prompts
        ]

        # Tokenize the final processed strings
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # TODO: check what should be the behavior, truncate or error or directly return?
        if input_ids.shape[-1] >= self.max_prompt_length and self.enforce_max_prompt_length:
            raise Exception(f"Prompt length {input_ids.shape[-1]} exceeds limit {self.max_prompt_length}")
        
        # pad to max sizes
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True
        )
        attention_mask = pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.max_prompt_length,
            pad_token_id=0,
            left_pad=True
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)

        # original_batch contains the extra info needed for generation
        if "meta_info" in kwargs and kwargs["meta_info"]:
            meta_info = kwargs["meta_info"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data
    
    def get_model_response_batched(self, prompts, all_dones, **kwargs):
        """Get model responses for non-done trajectories.
        
        Args:
            prompts: List of prompts for all agents
            all_dones: List of boolean flags indicating which trajectories are done
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Tuple of (responses, seq_idxs) where seq_idxs are the indices of non-done trajectories
        """
        seq_idxs = []
        cur_prompts = []
        for i, done in enumerate(all_dones):
            if not done:
                cur_prompts.append(prompts[i])
                seq_idxs.append(i)

        if cur_prompts:
            responses, seq_idxs = self.get_model_response(
                cur_prompts,
                seq_idxs,
                **kwargs,
            )
            assert len(responses) == len(seq_idxs), (
                f"Number of responses {len(responses)} returned does not match "
                f"number of trajectories {len(seq_idxs)}"
            )
            return responses, seq_idxs
        return [], []

    def step_environment_batched(self, actions, seq_idxs):
        """Step multiple environments in parallel.
        
        Args:
            actions: List of actions to take in each environment
            seq_idxs: List of indices indicating which environment each action is for
            
        Returns:
            Tuple of (next_observations, rewards, dones, infos)
        """
        results = [None] * len(seq_idxs)

        def step_env_single(env, action):
            return env.step(action)

        if all(type(self.envs[seq_idxs[i]]).is_multithread_safe() for i in range(len(seq_idxs))):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(step_env_single, self.envs[seq_idxs[i]], actions[i])
                    for i in range(len(seq_idxs))
                ]

                for i, fut in enumerate(futures):
                    results[i] = fut.result()
        else:
            for i in range(len(seq_idxs)):
                results[i] = self.envs[seq_idxs[i]].step(actions[i])

        next_observations, rewards, dones, infos = zip(*results)
        return list(next_observations), list(rewards), list(dones), list(infos)

    def reset_environment_batched(self, seq_idxs):
        """Reset multiple environments in parallel.
        
        Args:
            seq_idxs: List of indices indicating which environments to reset
            
        Returns:
            Tuple of (observations, infos)
        """
        results = [None] * len(seq_idxs)

        def reset_env_single(env):
            return env.reset()
        
        if all(type(self.envs[seq_idxs[i]]).is_multithread_safe() for i in range(len(seq_idxs))):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(reset_env_single, self.envs[seq_idxs[i]])
                    for i in range(len(seq_idxs))
                ]

                for i, fut in enumerate(futures):
                    results[i] = fut.result()
        else:
            for i in range(len(seq_idxs)):
                results[i] = self.envs[seq_idxs[i]].reset()

        # Unpack all results
        observations, infos = zip(*results)
        for info in infos:
            info['max_steps'] = self.max_steps
        return list(observations), list(infos)

    def close_environment_batched(self):
        """Close multiple environments in parallel."""
        def close_env_single(env):
            return env.close()
        

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(close_env_single, self.envs[i])
                for i in range(len(self.envs))
            ]
            # Wait for all futures to complete
            for fut in futures:
                fut.result() 

    def generate_trajectories(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        """
        Execute batched interactions with the environment and collect trajectories.

        This function simulates multiple parallel environments and collects step-wise 
        interactions in a structured format. Each trajectory consists of multiple 
        time steps, recording observations, actions, rewards, and other relevant 
        environment information.

        Args:
            reset_seed (int, optional): The random seed for resetting the environment. Defaults to 0.
            timing_raw (dict, optional): Dictionary for tracking execution times of different stages. Defaults to {}.
            mode (str, optional): Decides the return value. If `Text`,returns structured text. If 'Token', returns tokenized responses. If 'Conversation', returns the conversation format. Defaults to Text.
            **kwargs: Additional arguments that may be passed to environment interactions.

        Returns:
            List[List[Dict]]: A nested list where each inner list represents a trajectory 
            corresponding to a specific environment. Each step in the trajectory is 
            represented as a dictionary with the following keys:
            
            - "observation" (Any): The state observed before taking an action.
            - "next_observation" (Any): The state observed after taking an action.
            - "reward" (float): The reward received at this step.
            - "done" (bool): Whether the episode has ended.
            - "action" (Any): The action taken at this step.
            - "response" (str): The assistant's response.
            - "training_reward" (float): The computed reward signal for training purposes.
            - "truncated" (bool): If this end step resulted in max_steps exceed

            or

            List[Dict]: A list of dictionary where each represents the token version of the trajectory if `mode=Tokens`
            Each dict of the list has the following keys:

            - "prompt_tokens" (torch.tensor, dtype=torch.long): the prompt token obtained with initial observation
            - "response_tokens" (torch.tensor, dtype=torch.long): the response token obtained with the rest of response, next_observation of the trajectory
            - "response_masks" (torch.tensor, dtype=torch.long): the state masking for the response_tokens
            - "training_reward" (float): The computed reward signal for training purposes
            - "environment_reward" (float): The raw reward signal from the environment
            
            or

            List[List[Dict]]: A nested list where each inner list represents a trajectory conversation message in ChatML format 

        Notes:
            - The function ensures that trajectories remain in order, matching the environments they originated from.
            - The `mode` flag controls the return format.
            - Timing information, if provided via `timing_raw`, can be used for profiling execution times of different stages.
            - Trajectories content in the Token version has truncation due to max_response_length reflected, but not in Text or Conversation.
        """
        assert self.envs, f"Env cannot be empty, but got {self.envs}"
        env_batch_size = len(self.envs)
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."
        assert mode in ["Text", "Token", "Conversation", "Step"], f"Return mode {mode} not supported"

        timing_raw = timing_raw or {}

        for _ in range(self.retry_limit):
            try:
                steps = 0
                
                all_dones = [False for _ in range(env_batch_size)]
                max_prompt_token_len = 0
                all_response_token_lens = [0 for _ in range(env_batch_size)]
                all_prompt_tokens = [[] for _ in range(env_batch_size)]
                all_response_tokens = [[] for _ in range(env_batch_size)]
                all_response_masks = [[] for _ in range(env_batch_size)]
                
                 # For each episode, accumulate the prompt and response pair for each step in a list. 
                 # Each step is a dict with keys "prompt" and "response" to String
                all_steps = [[] for _ in range(env_batch_size)]

                observations, infos = self.reset_environment_batched(list(range(env_batch_size))) 

                # put initial observation into the sequence
                for i, obs in enumerate(observations):
                    self.agents[i].reset()
                    self.agents[i].update_from_env(
                        observation=obs,
                        reward=0,
                        done=False,
                        info=infos[i],
                    )
                    prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                        self.agents[i].chat_completions,
                        self.tokenizer,
                        self.chat_template_parser,
                        contains_first_msg=True,
                        contains_generation_msg=True
                    )
                    max_prompt_token_len = max(max_prompt_token_len, len(prompt_tokens))
                    all_prompt_tokens[i] = prompt_tokens
                    
                if max_prompt_token_len > self.max_prompt_length:
                    self.reset_agents_batched()
                    self.close_environment_batched()
                    raise Exception(f"Initial prompt length already exceeded max_prompt_length. Please set `max_prompt_length` to be larger. Current max_prompt_length is {max_prompt_token_len} and max_prompt_length is {self.max_prompt_length}.")
                    
                # get model actions and responses
                while not all(all_dones) and steps < self.max_steps:
                    steps += 1
                    prompt_response_pair = {}
                    with _timer("get_actions", timing_raw):
                        prompts = [self.agents[i].chat_completions.copy() for i in range(env_batch_size)]
                        # for enforced max prompt, no need to deduct here
                        if not self.enforce_max_prompt_length:
                            max_tokens = self.max_response_length - min([t for i, t in enumerate(all_response_token_lens) if all_dones[i] is False])
                        else:
                            max_tokens = self.max_response_length
                        kwargs['max_tokens'] = max_tokens
                        responses, seq_idxs = self.get_model_response_batched(
                            prompts, all_dones, **kwargs
                        )
                    
                    actions = []
                    for i, response in enumerate(responses):
                        self.agents[seq_idxs[i]].update_from_model(response)
                        cur_state = self.agents[seq_idxs[i]].get_current_state()
                        actions.append(cur_state.action)

                        prompt_response_pair = {
                            "prompt": self.chat_template_parser.parse(prompts[seq_idxs[i]], add_generation_prompt=True, is_first_msg=True),
                            "response": response,
                        }
                        all_steps[seq_idxs[i]].append(prompt_response_pair)

                    # environment step
                    try:
                        with _timer("env_step", timing_raw):
                            (
                                next_observations,
                                rewards,
                                dones,
                                infos,
                            ) = self.step_environment_batched(actions, seq_idxs)
                    except Exception as e:
                        print(f"Error in environment interation: {e}. Re-attempting...")
                        self.reset_agents()
                        raise e

                    for i, idx in enumerate(seq_idxs):
                        if all_dones[idx]:
                            raise Exception(f"Trajectory {idx} has new state but was marked done. Something went wrong.")
                        infos[i]['max_steps'] = self.max_steps
                        self.agents[idx].update_from_env(
                            observation=next_observations[i],
                            reward=rewards[i],
                            done=dones[i],
                            info=infos[i],
                        )

                        chat_completions_messages = self.agents[idx].chat_completions
                        assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

                        assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                            [assistant_message],
                            self.tokenizer,
                            self.chat_template_parser,
                            contains_first_msg=False,
                            contains_generation_msg=False
                        )
                        env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                            env_messages,
                            self.tokenizer,
                            self.chat_template_parser,
                            contains_first_msg=False,
                            contains_generation_msg=True
                        )

                        all_response_token_lens[idx] += len(assistant_msg_tokens) + len(env_msg_tokens)
                        # Reached maximum number of tokens for the trajectory
                        # If max prompt is enforced at each round, no need to enforce all response token length.
                        if not self.enforce_max_prompt_length and all_response_token_lens[idx] >= self.max_response_length:
                            # Truncation length
                            truncation_length = self.max_response_length - all_response_token_lens[idx]
                            # Truncate the response and masks
                            truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                            truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                            # Update token collections
                            all_response_tokens[idx].extend(truncated_response_tokens)
                            all_response_masks[idx].extend(truncated_response_masks)
                            all_dones[idx] = True

                            cur_step = self.agents[idx].get_current_state()
                            if all_response_token_lens[idx] - len(env_msg_tokens) > self.max_response_length:
                                cur_step.reward = 0.0
                            cur_step.done = True

                            colorful_print(
                                f"Trajectory {idx} completed due to maximum trajectory length reached. "
                                f"Reward is {rewards[i]}. \n",
                                "yellow",
                            )
                            continue
                        
                        # Update the token version of trajectory
                        all_response_tokens[idx].extend(assistant_msg_tokens)
                        all_response_masks[idx].extend(assistant_msg_masks)
                        observations[idx] = next_observations[i]
                        # If an environment is done, handle the completed trajectory
                        if dones[i]:
                            all_dones[idx] = True
                            colorful_print(
                                f"Trajectory {idx} completed. Reward is {rewards[i]}. \n",
                                "green",
                            )
                            continue
                        
                        # Insert env tokens to the results (only for non-finished trajectories)
                        all_response_tokens[idx].extend(env_msg_tokens)
                        all_response_masks[idx].extend(env_msg_masks)
                break

            except Exception as e:
                print(f"Error in environment interaction")
                print(traceback.format_exc())
                print(e)
                continue
        
        # Close all environments
        self.close_environment_batched()

        # Trajectory post-processing.
        trajectories = [a.trajectory for a in self.agents]
        for i, trajectory in enumerate(trajectories):
            compute_trajectory_reward(trajectory)
            compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectories
        elif mode == "Token":
            # Collect into dictionary form
            token_result = []
            for i, (prompt_tokens, response_tokens, response_masks) in enumerate(
                zip(all_prompt_tokens, all_response_tokens, all_response_masks)
            ):
                token_result.append({
                    "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                    "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                    "response_masks": torch.tensor(response_masks, dtype=torch.long),
                    "training_reward": self.agents[i].compute_training_reward(trajectories[i]) if hasattr(self.agents[i], "compute_training_reward") else trajectories[i].steps[-1].reward,
                    "environment_reward": trajectories[i].reward,
                    "idx": self.envs[i].idx,
                })
            return token_result
        elif mode == "Conversation":
            return [a.chat_completions for a in self.agents]
        elif mode == "Step":
            steps_result = []
            for i, episode in enumerate(all_steps):
                trajectory = trajectories[i]
                environment_reward = trajectory.reward
                training_reward =  self.agents[i].compute_training_reward(trajectory) if hasattr(self.agents[i], "compute_training_reward") else trajectory.steps[-1].reward
                steps_result.append({
                    "steps": episode,
                    "training_reward": training_reward,
                    "environment_reward": environment_reward,
                    "idx": self.envs[i].idx
                })
            return steps_result

    def reset_agents_batched(self):
        """Reset all agents in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(agent.reset)
                for agent in self.agents
            ]
            for future in futures:
                future.result()

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), (
            f"Number of agents must equal to number of environments but received, "
            f"{len(agents)} and {len(envs)}"
        )
        self.n_parallel_agents = len(envs)
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        
