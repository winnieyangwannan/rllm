import openai
import time
import os
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import threading
import asyncio
from queue import Queue, Empty
import torch

from verl.trainer.ppo.ray_trainer import _timer

from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward, compute_training_score, compute_environment_score


class BatchAgent:

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        agent_class,
        model_path="",
        n_parallel_agents=1,
        api_key=None,
        api_retries=3,
        env=None,
        gamma=0.95,
        retry_limit=5,
        episode_len=5,
        max_trajectory_length=8000,
        agent_args={},
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.api_key = api_key
        self.api_retries = api_retries
        self.agent_class = agent_class
        self.sampling_params = kwargs.get("sampling_params", None)
        self.n_parallel_agents = n_parallel_agents
        self.model_path = model_path

        self.agents = [agent_class(**agent_args)for _ in range(n_parallel_agents)]

        # For interaction
        self.env = env
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.episode_len = episode_len
        self.agent_args = agent_args
        self.max_trajectory_length = max_trajectory_length

    def get_actions(self, trajectories, seq_idxs=[], **kwargs):
        """
        Return a list of actions with same size as the trajectories list
        seq_idxs: The list of indexes that the trajectories are from. Used to index into self.agents

        return: Tuple (List of actions, List of responses)
        """
        if seq_idxs:
            assert len(trajectories) == len(
                seq_idxs
            ), f"Number of sequences {len(trajectories)} should equal to the number of agents they are for ({len(seq_idxs)})"

        if self.engine_name == "verl":
            return self._get_actions_verl(trajectories, seq_idxs, **kwargs)
        elif self.engine_name == "vllm":
            return self._get_actions_vllm(trajectories, seq_idxs, **kwargs)
        elif self.engine_name == "openai":
            return self._get_actions_openai(trajectories, seq_idxs, **kwargs)
        else:
            raise NotImplementedError

    def _get_actions_verl(self, trajectories, seq_idxs, **kwargs):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(traj)
            for i, traj in enumerate(trajectories)
        ]

        batch = self._convert_prompt_verl(prompts, **kwargs)

        # because of veRL's chunking. we need to pad number of prompts to be a multiple of worker group world size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.rollout_engine.world_size)
        output_padded = self.rollout_engine.generate_sequences(batch_padded)
        
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        output_text = self.tokenizer.batch_decode(
            output.batch["responses"], skip_special_tokens=False
        )

        pad_token = self.tokenizer.pad_token
        responses = []
        for i, text in enumerate(output_text):
            rsp = text.replace(pad_token, "")
            responses.append(rsp)

        assert len(responses) == len(
            trajectories
        ), f"Number of responses {len(responses)} should equal to the number of trajectories ({len(trajectories)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(trajectories))
        ]
        return actions, responses

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL
        """
        from verl.utils.model import compute_position_id_with_mask
        from verl import DataProto
        from verl.protocol import union_two_dict

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
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

    def _get_actions_vllm(self, trajectories, seq_idxs, **kwargs):
        # Format each observation into a prompt
        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(traj)
            for i, traj in enumerate(trajectories)
        ]

        prompts_token = self.tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=False
        )

        # Generate responses using vLLM
        outputs = self.rollout_engine.generate(
            prompts=prompts_token, sampling_params=self.sampling_params, use_tqdm=False
        )

        # Decode the output token IDs into text
        responses = []
        # Get the generated text directly from the RequestOutput object
        for i, output in enumerate(outputs):
            rsp = output.outputs[0].text
            responses.append(rsp)

        assert len(responses) == len(
            trajectories
        ), f"Number of responses {len(responses)} should equal to the number of trajectories ({len(trajectories)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(trajectories))
        ]
        return actions, responses


    def _get_actions_openai(self, trajectories, seq_idxs, **kwargs):
        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(traj)
            for i, traj in enumerate(trajectories)
        ]

        openai.api_key = self.api_key
        responses = []
        with mp.Pool(os.cpu_count()) as pool:
            for i, response in enumerate(
                tqdm(pool.imap(self._get_openai_response, prompts), total=len(prompts))
            ):
                responses.append(response)

        assert len(responses) == len(
            trajectories
        ), f"Number of responses {len(responses)} should equal to the number of trajectories ({len(trajectories)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(trajectories))
        ]
        return actions, responses


    def _get_openai_response(self, prompt):
        # GPT prompt
        retries = self.api_retries
        while retries > 0:
            try:
                # OpenAI API call
                response = openai.chat.completions.create(
                    model="o1-preview", messages=prompt
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                print(f"Sleep for 5 seconds for API limit.")
                time.sleep(5)
            except Exception as e:
                return f"Error processing content: {e}"

    # deals with the case when the trajectory is done
    def _safe_get_actions(self, trajectories, batch_done, **kwargs):
        new_trajectory_sequences = []
        seq_idxs = []
        responses = [""] * len(trajectories)
        actions = [""]*len(trajectories)

        for i, done in enumerate(batch_done):
            if not done and trajectories[i][-1]['next_observation'] is not None:
                new_trajectory_sequences.append(trajectories[i])
                seq_idxs.append(i)

        if len(new_trajectory_sequences) > 0:
            gen_actions, gen_responses = self.get_actions(
                new_trajectory_sequences,
                seq_idxs,
                **kwargs,
            )

            for i, idx in enumerate(seq_idxs):
                actions[idx] = gen_actions[i].replace("<|im_end|>", "")
                responses[idx] = gen_responses[i].replace("<|im_end|>", "")
                
        for action, traj in zip(actions, new_trajectory_sequences):
            new_obs = traj[-1]['next_observation']
            if new_obs is None:
                assert action == "", "Action should be empty, First assert"

        return actions, responses

    def interact_environment(self, reset_seed=0, timing_raw={}, mode="Text", **kwargs):
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
            - "truncated" (bool): If this end step resulted in max_episode_length exceed

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
            - Trajectories content in the Token version has truncation due to max_trajectory_length reflected, but not in Text or Conversation.

        """
        assert self.env, f"Env cannot be empty, but got {self.env}"
        assert hasattr(self.env, "batch_size"), "Env does not have batch_size attribute"
        env_batch_size = self.env.batch_size
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."
        assert mode in ["Text", "Token", "Conversation"], f"Return mode {mode} not supported"

        trajectories = [[] for _ in range(env_batch_size)]

        for _ in range(self.retry_limit):
            try:
                done = False
                trajectories = [[] for _ in range(env_batch_size)]

                steps = 0
                observations, infos = self.env.reset(seed=reset_seed)
                batch_done = [False for _ in range(env_batch_size)]

                # For veRL training
                max_prompt_token_len = 0
                all_response_token_lens = [0 for _ in range(env_batch_size)]
                all_prompt_tokens = [[] for _ in range(env_batch_size)]
                all_response_tokens = [[] for _ in range(env_batch_size)]
                all_response_masks = [[] for _ in range(env_batch_size)]

                # For returning conversations
                all_conversations = [[] for _ in range(env_batch_size)]

                # put initial observation into the sequence
                for i, obs in enumerate(observations):
                    trajectories[i].append({"next_observation": obs})

                    # compute initial prompt tokens
                    initial_msg = {
                        "role": "user",
                        "content": self.agents[i].convert_observation_to_string(obs, with_system_prompt=True),
                    }
                    prompt_tokens, _ = self._convert_message_to_tokens_and_masks(initial_msg)

                    max_prompt_token_len = max(max_prompt_token_len, len(prompt_tokens))
                    all_prompt_tokens[i] = prompt_tokens

                    # Update conversation version
                    all_conversations[i].append(initial_msg)

                if max_prompt_token_len >= self.max_trajectory_length:
                    self.reset()
                    raise Exception("Initial prompt length already exceeded max_trajectory_length, retrying")
                    
                # get model actions and responses
                while not all(batch_done) and steps < self.episode_len:
                    steps += 1
                    with _timer("get_actions", timing_raw):
                        actions, responses = self._safe_get_actions(
                            trajectories, batch_done, **kwargs
                        )

                    for action, done in zip(actions, batch_done):
                        if done:
                            assert action == ""

                    # environment step
                    try:
                        with _timer("env_step", timing_raw):
                            (
                                next_observations,
                                rewards,
                                terminateds,
                                truncateds,
                                infos,
                            ) = self.env.step(actions)
                    except Exception as e:
                        print(e)
                        self.reset()

                    colorful_print(
                        f"Step {steps} in environment interation done. {len(actions)} actions generated. responses: {responses}, actions: {actions}\n",
                        "green",
                    )

                    for i in range(env_batch_size):
                        if batch_done[i]:
                            continue

                        # Update the agent
                        self.agents[i].update(
                            actions[i],
                            observations[i],
                            next_observations[i],
                            rewards[i],
                            terminateds[i],
                            truncateds[i],
                            infos[i],
                        )


                        # Compute the response tokens and response masks for the trajectory
                        assistant_msg = {"role": "assistant", "content": responses[i]}

                        next_obs = next_observations[i]
                        next_obs_txt = self.agents[i].convert_observation_to_string(next_obs, with_system_prompt=False)
                        env_msg = {"role": "user", "content": next_obs_txt}

                        assistant_msg_tokens, assistant_msg_masks = self._convert_message_to_tokens_and_masks(assistant_msg)
                        env_msg_tokens, env_msg_masks = self._convert_message_to_tokens_and_masks(env_msg)

                        # Reached maximum number of tokens for the trajectory
                        if all_response_token_lens[i] + len(assistant_msg_tokens) + len(env_msg_tokens) + max_prompt_token_len >= self.max_trajectory_length:
                            # Truncation length
                            truncation_length = self.max_trajectory_length - max_prompt_token_len - all_response_token_lens[i]
                            # Truncate the response and masks
                            truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                            truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                            # Update the token version of trajectory (Though it is truncated)
                            all_response_tokens[i].extend(truncated_response_tokens)
                            all_response_masks[i].extend(truncated_response_masks)
                            # Update conversation (Though it is truncated)
                            all_conversations[i].append(assistant_msg)
                            all_conversations[i].append(env_msg)
                            # Update trajectory (Though it is truncated)
                            trajectories[i].append(
                                {
                                    "observation": observations[i],
                                    "next_observation": next_observations[i],
                                    "reward": 0, # TODO: May need to update this to be minimum environment score in the future
                                    "done": terminateds[i] or truncateds[i],
                                    "action": actions[i],
                                    "info": infos[i],
                                    "response": responses[i],
                                    "truncated": True,
                                }
                            )
                            batch_done[i] = True
                            colorful_print(
                                f"Trajectory {i} completed due to maximum trajectory length reached. But entire Text or Conversation will be returned. Reward is {rewards[i]}. \n",
                                "yellow",
                            )
                            continue

                        # If an environment is done, handle the completed trajectory
                        if terminateds[i] or truncateds[i]:
                            batch_done[i] = True
                            colorful_print(
                        f"Trajectory {i} completed due to {'terminaion' if terminateds[i] else 'truncation'}. Reward is {rewards[i]}. \n",
                                "green",
                            )

                        # Update repsonse token length
                        all_response_token_lens[i] += len(assistant_msg_tokens) + len(env_msg_tokens)

                        # Update the token version of trajectory
                        all_response_tokens[i].extend(assistant_msg_tokens + env_msg_tokens)
                        all_response_masks[i].extend(assistant_msg_masks + env_msg_masks)

                        # Update conversation version
                        all_conversations[i].append(assistant_msg)
                        all_conversations[i].append(env_msg)

                        # Update the trajectory
                        trajectories[i].append(
                            {
                                "observation": observations[i],
                                "next_observation": next_observations[i],
                                "reward": rewards[i],
                                "done": terminateds[i] or truncateds[i],
                                "action": actions[i],
                                "info": infos[i],
                                "response": responses[i],
                                "truncated": False,
                            }
                        )
                        observations[i] = next_observations[i]

                break

            except Exception as e:
                print(f"Error in environment interaction")
                import traceback

                print(traceback.format_exc())
                print(e)
                continue
        
        # remove sentinel
        trajectories = [traj[1:] if traj else [] for traj in trajectories]
        trajectory_result = []

        for i, trajectory in enumerate(trajectories):
            augmented_trajectory = add_mc_return(add_trajectory_reward(trajectory), gamma=self.gamma)
            training_reward = self.agents[i].compute_training_reward(augmented_trajectory)
            trajectory_result.append(add_training_reward(augmented_trajectory, training_reward))

        if mode == "Text":
            return trajectory_result
        
        if mode == "Token":
            # Collect into dictionary form
            token_result = []
            for i, (prompt_tokens, response_tokens, response_masks) in enumerate(zip(all_prompt_tokens, all_response_tokens, all_response_masks)):
                trajectory = trajectory_result[i]
                training_reward = compute_training_score(trajectory)
                env_reward = compute_environment_score(trajectory)

                token_result.append({
                    "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                    "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                    "response_masks": torch.tensor(response_masks, dtype=torch.long),
                    "training_reward": training_reward,
                    "environment_reward": env_reward,
                })
            return token_result

        if mode == "Conversation":
            return all_conversations
        

    def reset(self):
        """
        Resets all agents.
        """
        for agent in self.agents:
            agent.reset()

    def update_env(self, env):
        """
        Updates the environment and recreates all agents. Instead of reusing existing agents, this method creates new agent instances to match 
        the batch size of the new environment.

        Args:
            env: The new environment instance to be assigned.
        """
        self.n_parallel_agents = env.batch_size

        self.agents = [self.agent_class(**self.agent_args)for _ in range(self.n_parallel_agents)]

        self.env = env

    def _postprocess_model_chat_template(self, message_text):
        """
        Postprocesses the chat template output by removing any automatically added system message.

        Args:
            message_text (str): The formatted message text.

        Returns:
            str: The processed message text without the default system message.
        """
        if any(substring in self.model_path.lower() for substring in ('qwen2', 'qwen2.5')):
            # from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json, a default system message is inserted. So we manually remove the first occurance of default system message.
            # This is currently assuming no tool call.
            target = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            if message_text.startswith(target):
                return message_text[len(target):]  # Remove only if it’s at the start
            return message_text

        print("Model not recognized for postprocessing, entire text returned")
        return message_text
    
    def _convert_message_to_tokens_and_masks(self, msg):
        msg_text = self.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )
        msg_text = self._postprocess_model_chat_template(msg_text)

        msg_tokens = self.tokenizer.encode(msg_text, add_special_tokens=False)

        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        return msg_tokens, msg_mask