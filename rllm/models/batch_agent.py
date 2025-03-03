import openai
import time
import os
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import threading
import asyncio
from queue import Queue, Empty

from verl.trainer.ppo.ray_trainer import _timer

from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward


class BatchAgent:

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        agent_class,
        n_parallel_agents=1,
        api_key=None,
        api_retries=3,
        env=None,
        gamma=0.95,
        retry_limit=5,
        episode_len=5,
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

        self.agents = [agent_class(**agent_args)for _ in range(n_parallel_agents)]

        # For interaction
        self.env = env
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.episode_len = episode_len
        self.agent_args = agent_args

    def get_actions(self, obs_action_sequences, seq_idxs=[], **kwargs):
        """
        Return a list of actions with same size as the obs_action_sequences list
        seq_idxs: The list of indexes that the obs_action_sequences are from. Used to index into self.agents

        return: Tuple (List of actions, List of responses)
        """
        if seq_idxs:
            assert len(obs_action_sequences) == len(
                seq_idxs
            ), f"Number of sequences {len(obs_action_sequences)} should equal to the number of agents they are for ({len(seq_idxs)})"

        if self.engine_name == "verl":
            return self._get_actions_verl(obs_action_sequences, seq_idxs, **kwargs)
        elif self.engine_name == "vllm":
            return self._get_actions_vllm(obs_action_sequences, seq_idxs, **kwargs)
        elif self.engine_name == "openai":
            return self._get_actions_openai(obs_action_sequences, seq_idxs, **kwargs)
        else:
            raise NotImplementedError

    def _get_actions_verl(self, obs_action_sequences, seq_idxs, **kwargs):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(obs_act_seq)
            for i, obs_act_seq in enumerate(obs_action_sequences)
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
            obs_action_sequences
        ), f"Number of responses {len(responses)} should equal to the number of obs_action_sequences ({len(obs_action_sequences)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(obs_action_sequences))
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

    def _get_actions_vllm(self, obs_action_sequences, seq_idxs, **kwargs):
        # Format each observation into a prompt
        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(obs_act_seq)
            for i, obs_act_seq in enumerate(obs_action_sequences)
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
            obs_action_sequences
        ), f"Number of responses {len(responses)} should equal to the number of obs_action_sequences ({len(obs_action_sequences)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(obs_action_sequences))
        ]
        return actions, responses


    def _get_actions_openai(self, obs_action_sequences, seq_idxs, **kwargs):
        prompts = [
            self.agents[seq_idxs[i]]._pre_get_action(obs_act_seq)
            for i, obs_act_seq in enumerate(obs_action_sequences)
        ]

        openai.api_key = self.api_key
        responses = []
        with mp.Pool(os.cpu_count()) as pool:
            for i, response in enumerate(
                tqdm(pool.imap(self._get_openai_response, prompts), total=len(prompts))
            ):
                responses.append(response)

        assert len(responses) == len(
            obs_action_sequences
        ), f"Number of responses {len(responses)} should equal to the number of obs_action_sequences ({len(obs_action_sequences)})"

        actions = [
            self.agents[seq_idxs[i]]._post_get_action(responses[i])
            for i in range(len(obs_action_sequences))
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
    def _safe_get_actions(self, obs_action_sequences, batch_done, **kwargs):
        new_obs_action_sequences = []
        seq_idxs = []
        responses = [""] * len(obs_action_sequences)
        actions = [""]*len(obs_action_sequences)

        for i, done in enumerate(batch_done):
            if not done and obs_action_sequences[i][-1] is not None:
                new_obs_action_sequences.append(obs_action_sequences[i])
                seq_idxs.append(i)

        if len(new_obs_action_sequences) > 0:
            gen_actions, gen_responses = self.get_actions(
                new_obs_action_sequences,
                seq_idxs,
                **kwargs,
            )

            for i, idx in enumerate(seq_idxs):
                actions[idx] = gen_actions[i].replace("<|im_end|>", "")
                responses[idx] = gen_responses[i].replace("<|im_end|>", "")
                
        for action, obs_act_seq in zip(actions, obs_action_sequences):
            new_obs = obs_act_seq[-1]
            if new_obs is None:
                assert action == "", "Action should be empty, First assert"

        return actions, responses

    def interact_environment(self, reset_seed=0, timing_raw={}, **kwargs):
        """
        Run environment interactions in batches.
        Collects trajectories in the format:
        [[{"observation":, "next_observation":, "reward":, "done":, "action":, "response":, "training_reward": },...],...]

        Returned trajectories are in order of the environments they are from.
        """
        assert self.env, f"Env cannot be empty, but got {self.env}"
        assert hasattr(self.env, "batch_size"), "Env does not have batch_size attribute"
        env_batch_size = self.env.batch_size
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."

        trajectories = [[] for _ in range(env_batch_size)]

        for _ in range(self.retry_limit):
            try:
                done = False
                trajectories = [[] for _ in range(env_batch_size)]
                obs_action_sequences = [[] for _ in range(env_batch_size)]

                steps = 0
                observations, infos = self.env.reset(seed=reset_seed)
                batch_done = [False for _ in range(env_batch_size)]

                # put initial observation into the sequence
                for i, obs in enumerate(observations):
                    obs_action_sequences[i].append(obs)

                # get model actions and responses
                while not all(batch_done) and steps < self.episode_len:
                    steps += 1
                    with _timer("get_actions", timing_raw):
                        actions, responses = self._safe_get_actions(
                            obs_action_sequences, batch_done, **kwargs
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
                        raise (e)

                    colorful_print(
                        f"Step {steps} in environment interation done. {len(actions)} actions generated. responses: {responses} \n",
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

                        # Add response and next observation for next round of generation
                        obs_action_sequences[i].append(responses[i])
                        obs_action_sequences[i].append(next_observations[i])

                        # If an environment is done, handle the completed trajectory
                        if terminateds[i] or truncateds[i]:
                            batch_done[i] = True
                            colorful_print(
                                f"Trajectory {i} completed due to {'termination' if terminateds[i] else 'truncation'}. Reward is {rewards[i]}. \n",
                                "green",
                            )

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

        result = []

        for i, trajectory in enumerate(trajectories):
            augmented_trajectory = add_mc_return(add_trajectory_reward(trajectory), gamma=self.gamma)
            training_reward = self.agents[i].compute_training_reward(augmented_trajectory)
            result.append(add_training_reward(augmented_trajectory, training_reward))

        return result

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