import openai
import time
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch

from verl.trainer.ppo.ray_trainer import _timer

from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward, compute_training_score, compute_environment_score


class AgentExecutionEngine:

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        agents=[], # List of agents
        envs=[], # List of environments
        model_path="",
        gamma=0.95,
        api_retries=3,
        retry_limit=1,
        max_episodes=5,
        max_prompt_length=512, # Max prompt length for agent is only applied to first request, all subsequent requests are considered to be results.
        max_trajectory_length=8000,
        rollout_engine_args={},
        **kwargs,
    ):
        assert max_trajectory_length > max_prompt_length, f"Max trajectory length {max_trajectory_length} must be greater than max prompt length {max_prompt_length}."
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = len(envs)
        self.model_path = model_path

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_episodes = max_episodes
        self.max_trajectory_length = max_trajectory_length
        self.max_prompt_length = max_prompt_length

        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.agents = agents
        self.envs = envs

        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", None)

        if engine_name == "openai":
            from openai import OpenAI 
            self.client = OpenAI(**self.rollout_engine_args)

    def get_actions(self, trajectories, seq_idxs, **kwargs):
        """
        Return a list of actions with same size as the trajectories list
        seq_idxs: The list of indexes that the trajectories are from. Used to index into self.agents and self.envs

        return: Tuple (List of actions, List of responses)
        """
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

        # print("messages:", prompts)

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

        # Use ThreadPoolExecutor instead of multiprocessing
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(tqdm(executor.map(self.call_openai, prompts), total=len(prompts)))

        assert len(responses) == len(
            trajectories
        ), f"Number of responses {len(responses)} should equal to the number of trajectories ({len(trajectories)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(trajectories))
        ]
        return actions, responses


    def _call_openai(self, prompt):
        retries = self.api_retries
        while retries > 0:
            try:
                response = self.client.chat.completions.create(
                    model="o1-preview", messages=prompt
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                time.sleep(5)
            except Exception as e:
                return f"Error processing content: {e}"

    # deals with the case when the trajectory is done
    def _safe_get_actions(self, trajectories, batch_done, **kwargs):
        new_trajectory_sequences = []
        seq_idxs = []
        responses = []
        actions = []

        for i, done in enumerate(batch_done):
            if not done:
                assert trajectories[i][-1]['next_observation'] is not None, f"Something went wrong, newest observation is None when trajectory hasn't terminated, index {i}"
                new_trajectory_sequences.append(trajectories[i])
                seq_idxs.append(i)

        if len(new_trajectory_sequences) > 0:
            gen_actions, gen_responses = self.get_actions(
                new_trajectory_sequences,
                seq_idxs,
                **kwargs,
            )
            assert len(gen_actions) == len(seq_idxs), f"Number of actions {len(gen_actions)} returned does not match number of trajectories {len(seq_idxs)}"
            # ToDO (Sijun): What is the purpose of this?
            for i, idx in enumerate(seq_idxs):
                if isinstance(gen_actions[i], str):
                    actions.append(gen_actions[i].replace("<|im_end|>", ""))
                else:
                    actions.append(gen_actions[i])
                responses.append(gen_responses[i].replace("<|im_end|>", ""))

        return actions, responses, seq_idxs

    def step_env_single(self, env, action):
        return env.step(action)

    def step_environment(self, actions, seq_idxs):
        results = [None] * len(seq_idxs)

        with ThreadPoolExecutor(max_workers=len(seq_idxs)) as executor:
            futures = [
                executor.submit(self.step_env_single, self.envs[seq_idxs[i]], actions[i])
                for i in range(len(seq_idxs))
            ]

            for i, fut in enumerate(futures):
                results[i] = fut.result()

        # Unpack all results
        next_observations, rewards, terminateds, truncateds, infos = zip(*results)
        return list(next_observations), list(rewards), list(terminateds), list(truncateds), list(infos)
    
    def reset_env_single(self, env, seed=0):
        return env.reset(seed=seed)

    def reset_environment(self, seq_idxs, seed=0):
        results = [None] * len(seq_idxs)

        with ThreadPoolExecutor(max_workers=len(seq_idxs)) as executor:
            futures = [
                executor.submit(self.reset_env_single, self.envs[seq_idxs[i]], seed=seed)
                for i in range(len(seq_idxs))
            ]

            for i, fut in enumerate(futures):
                results[i] = fut.result()

        # Unpack all results
        observations, infos = zip(*results)
        return list(observations), list(infos)

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
            - "truncated" (bool): If this end step resulted in max_episodes exceed

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
        assert self.envs, f"Env cannot be empty, but got {self.envs}"
        env_batch_size = len(self.envs)
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."
        assert mode in ["Text", "Token", "Conversation"], f"Return mode {mode} not supported"

        trajectories = [[] for _ in range(env_batch_size)]

        for _ in range(self.retry_limit):
            try:
                trajectories = [[] for _ in range(env_batch_size)]

                steps = 0
                observations, infos = self.reset_environment(list(range(env_batch_size)), seed=reset_seed)
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

                    initial_messages = self.agents[i].format_observation_as_messages(obs)
                    prompt_tokens, _ = self._convert_messages_to_tokens_and_masks(initial_messages)

                    max_prompt_token_len = max(max_prompt_token_len, len(prompt_tokens))
                    all_prompt_tokens[i] = prompt_tokens

                    # Update conversation version
                    all_conversations[i].extend(initial_messages)

                if max_prompt_token_len > self.max_prompt_length:
                    self.reset()
                    raise Exception("Initial prompt length already exceeded max_prompt_length, retrying")
                max_prompt_token_len = self.max_prompt_length
                    
                # get model actions and responses
                while not all(batch_done) and steps < self.max_episodes:
                    steps += 1
                    with _timer("get_actions", timing_raw):
                        actions, responses, seq_idxs = self._safe_get_actions(
                            trajectories, batch_done, **kwargs
                        )

                    # environment step
                    try:
                        with _timer("env_step", timing_raw):
                            (
                                next_observations,
                                rewards,
                                terminateds,
                                truncateds,
                                infos,
                            ) = self.step_environment(actions, seq_idxs)
                    except Exception as e:
                        print(e)
                        self.reset()
                        raise(e)

                    colorful_print(
                        f"Step {steps} in environment interation done. {len(actions)} actions generated. responses: {responses}, actions: {actions}\n",
                        "green",
                    )

                    for i, idx in enumerate(seq_idxs):
                        if batch_done[idx]:
                            raise Exception(f"Trajectory has new state but is done. Something went wrong. Index {i}")

                        # Update the agent
                        self.agents[idx].update(
                            actions[i],
                            observations[idx],
                            next_observations[i],
                            rewards[i],
                            terminateds[i],
                            truncateds[i],
                            infos[i],
                        )
                        
                        # Compute the response tokens and response masks for the trajectory
                        assistant_msg = {"role": "assistant", "content": responses[i]}

                        next_obs = next_observations[i]
                        env_messages = self.agents[idx].format_observation_as_messages(next_obs, with_system_prompt=False)
                        env_msg_tokens, env_msg_masks = self._convert_messages_to_tokens_and_masks(env_messages)

                        assistant_msg_tokens, assistant_msg_masks = self._convert_message_to_tokens_and_masks(assistant_msg)

                        # Reached maximum number of tokens for the trajectory
                        if all_response_token_lens[idx] + len(assistant_msg_tokens) + len(env_msg_tokens) + max_prompt_token_len >= self.max_trajectory_length:
                            # Truncation length
                            truncation_length = self.max_trajectory_length - max_prompt_token_len - all_response_token_lens[idx]
                            # Truncate the response and masks
                            truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                            truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                            # Update the token version of trajectory (Though it is truncated)
                            all_response_tokens[idx].extend(truncated_response_tokens)
                            all_response_masks[idx].extend(truncated_response_masks)
                            # Update conversation (Though it is truncated)
                            all_conversations[idx].append(assistant_msg)
                            all_conversations[idx].extend(env_messages)
                            # Update trajectory (Though it is truncated)
                            trajectories[idx].append(
                                {
                                    "observation": observations[idx],
                                    "next_observation": next_observations[i],
                                    "reward": 0, # TODO: May need to update this to be minimum environment score in the future
                                    "done": terminateds[i] or truncateds[i],
                                    "action": actions[i],
                                    "info": infos[i],
                                    "response": responses[i],
                                    "truncated": True,
                                }
                            )
                            batch_done[idx] = True
                            colorful_print(
                                f"Trajectory {idx} completed due to maximum trajectory length reached. But entire Text or Conversation will be returned. Reward is {rewards[i]}. \n",
                                "yellow",
                            )
                            continue

                        # If an environment is done, handle the completed trajectory
                        if terminateds[i] or truncateds[i]:
                            batch_done[idx] = True
                            colorful_print(
                                f"Trajectory {idx} completed due to {'terminaion' if terminateds[i] else 'truncation'}. Reward is {rewards[i]}. \n",
                                "green",
                            )

                        # Update repsonse token length
                        all_response_token_lens[idx] += len(assistant_msg_tokens) + len(env_msg_tokens)

                        # Update the token version of trajectory
                        all_response_tokens[idx].extend(assistant_msg_tokens + env_msg_tokens)
                        all_response_masks[idx].extend(assistant_msg_masks + env_msg_masks)

                        # Update conversation version
                        all_conversations[idx].append(assistant_msg)
                        all_conversations[idx].extend(env_messages)

                        # Update the trajectory
                        trajectories[idx].append(
                            {
                                "observation": observations[idx],
                                "next_observation": next_observations[i],
                                "reward": rewards[i],
                                "done": terminateds[i] or truncateds[i],
                                "action": actions[i],
                                "info": infos[i],
                                "response": responses[i],
                                "truncated": False,
                            }
                        )
                        observations[idx] = next_observations[i]

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

    def update_envs_and_agents(self, envs, agents):
        """
        Updates the environment and agent. 

        Args:
            envs: List of environments to use.
            agent: List of agents to use.
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.n_parallel_agents = len(envs)
        self.envs = envs
        self.agents = agents

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
    
    def _convert_messages_to_tokens_and_masks(self, messages):
        all_msg_tokens = []
        all_msg_masks = []

        for msg in messages:
            msg_tokens, msg_mask = self._convert_message_to_tokens_and_masks(msg)
            all_msg_tokens.extend(msg_tokens)
            all_msg_masks.extend(msg_mask)

        return all_msg_tokens, all_msg_masks
        