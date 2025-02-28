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
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return


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
        safe_batch_size=100,
        gamma=0.95,
        retry_limit=5,
        episode_len=2,
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.api_key = api_key
        self.api_retries = api_retries
        self.kwargs = kwargs
        self.agent_class = agent_class
        self.sampling_params = kwargs.get("sampling_params", None)
        self.n_parallel_agents = n_parallel_agents

        self.agents = [
            agent_class(
                rollout_engine=rollout_engine,
                engine_name=engine_name,
                tokenizer=tokenizer,
                api_key=api_key,
                api_retries=api_retries,
                **kwargs,
            )
            for _ in range(n_parallel_agents)
        ]

        # For interaction
        self.env = env
        self.safe_batch_size = safe_batch_size
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.episode_len = episode_len

    def get_actions(self, observations, obs_idxs=[], **kwargs):
        """
        Return a list of actions with same size as the observations list
        obs_idxs: The list of indexes that the observations are from. Used to index into self.agents

        return: Tuple (List of actions, List of responses)
        """
        if obs_idxs:
            assert len(observations) == len(
                obs_idxs
            ), f"Number of observations {len(observations)} should equal to the number of agents they are for ({len(obs_idxs)})"

        if self.engine_name == "verl":
            return self._get_actions_verl(observations, obs_idxs, **kwargs)
        elif self.engine_name == "vllm":
            return self._get_actions_vllm(observations, obs_idxs, **kwargs)
        elif self.engine_name == "openai":
            return self._get_actions_openai(observations, obs_idxs, **kwargs)
        else:
            raise NotImplementedError

    def _get_actions_verl(self, observations, obs_idxs=[], **kwargs):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        prompts = [
            self.agents[obs_idxs[i]]._pre_get_action(obs)
            for i, obs in enumerate(observations)
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
            observations
        ), f"Number of responses {len(responses)} should equal to the number of observations ({len(observations)})"

        actions = [
            self.agents[obs_idxs[i]]._post_get_action(responses[i])
            for i, obs in enumerate(observations)
        ]
        return actions, responses

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL
        """
        from verl.utils.model import compute_position_id_with_mask
        from verl import DataProto
        from verl.protocol import union_two_dict

        inputs = self.tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

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
        if "original_batch" in kwargs and kwargs["original_batch"]:
            original_batch = kwargs["original_batch"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, original_batch.meta_info)

        return data

    def _get_actions_vllm(self, observations, obs_idxs=[], **kwargs):
        # Format each observation into a prompt
        prompts = [
            self.agents[obs_idxs[i]]._pre_get_action(obs)
            for i, obs in enumerate(observations)
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
            observations
        ), f"Number of responses {len(responses)} should equal to the number of observations ({len(observations)})"

        actions = [
            self.agents[obs_idxs[i]]._post_get_action(responses[i])
            for i, obs in enumerate(observations)
        ]
        return actions, responses

    def _get_actions_openai(self, observations, obs_idxs=[], **kwargs):
        prompts = [
            self.agents[obs_idxs[i]]._pre_get_action(obs)
            for i, obs in enumerate(observations)
        ]

        openai.api_key = self.api_key
        responses = []
        with mp.Pool(os.cpu_count()) as pool:
            for i, response in enumerate(
                tqdm(pool.imap(self._get_openai_response, prompts), total=len(prompts))
            ):
                responses.append(response)

        assert len(responses) == len(
            observations
        ), f"Number of responses {len(responses)} should equal to the number of observations ({len(observations)})"

        actions = [
            self.agents[obs_idxs[i]]._post_get_action(responses[i])
            for i, obs in enumerate(observations)
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
    def _safe_get_actions(self, observations, batch_done, **kwargs):
        new_observations = []
        obs_idxs = []
        actions = [""] * len(observations)
        responses = [""] * len(observations)
        for i, done in enumerate(batch_done):
            if not done and observations[i] is not None:
                new_observations.append(observations[i])
                obs_idxs.append(i)

        if len(new_observations) > 0:
            gen_responses = []
            gen_actions = []
            for i in range(0, len(new_observations), self.safe_batch_size):
                try:
                    new_actions, new_responses = self.get_actions(
                        new_observations[i : i + self.safe_batch_size],
                        obs_idxs[i : i + self.safe_batch_size],
                        **kwargs,
                    )
                    gen_actions.extend(new_actions)
                    gen_responses.extend(new_responses)
                except Exception as e:
                    # sometime it says unable to infer image channel error
                    colorful_print(f"Error in getting action: {e}", "red")
                    # outputs += ["ERROR"]*len(new_observations[i:i+safe_batch_size])
                    raise e

            for i, idx in enumerate(obs_idxs):
                actions[idx] = gen_actions[i]
                responses[idx] = gen_responses[i]

        for action, obs in zip(actions, observations):
            if obs is None:
                assert action == "", "Action should be empty, First assert"

        return actions, responses

    def interact_environment(self, reset_seed=0, timing_raw={}, **kwargs):
        """
        Run environment interactions in batches.
        Collects trajectories in the format:
        [[{"observation":, "next_observation":, "reward":, "done":, "action": , "response": , "augmented_reward": },...],...]
        Returned trajectories are in order of the environments they are from.
        """
        assert self.env, f"Env cannot be empty, but got {self.env}"
        assert hasattr(self.env, "batch_size"), "Env does not have batch_size attribute"
        env_batch_size = self.env.batch_size
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."

        trajectories = []

        for k in range(self.retry_limit):
            try:
                done = False
                trajectories = [[] for _ in range(env_batch_size)]

                steps = 0
                observations, infos = self.env.reset(seed=reset_seed)
                batch_done = [False for _ in range(env_batch_size)]

                while not all(batch_done) and steps < self.episode_len:
                    steps += 1
                    with _timer("get_actions", timing_raw):
                        actions, responses = self._safe_get_actions(
                            observations, batch_done, **kwargs
                        )

                    for action, done in zip(actions, batch_done):
                        if done:
                            assert action == ""

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
                        f"Step {steps} in environment interation done. {len(actions)} actions generated. actions: {actions} \n",
                        "green",
                    )

                    for i in range(env_batch_size):
                        if batch_done[i]:
                            continue

                        aug_reward = self.agents[i].augment_reward(responses[i], next_observations[i], rewards[i])

                        self.agents[i].update(
                            actions[i],
                            observations[i],
                            next_observations[i],
                            rewards[i],
                            terminateds[i],
                            truncateds[i],
                            infos[i],
                        )

                        # If an environment is done, handle the completed trajectory
                        if terminateds[i] or truncateds[i]:
                            batch_done[i] = True
                            colorful_print(
                                f"Trajectory {i} completed due to {'termination' if terminateds[i] else 'truncation'}. Reward is {rewards[i]}. \n",
                                "green",
                            )
                       
                        trajectories[i].append(
                            {
                                "observation": observations[i],
                                "next_observation": next_observations[i],
                                "reward": rewards[i],
                                "done": terminateds[i] or truncateds[i],
                                "action": actions[i],
                                "info": infos[i],
                                "augmented_reward": aug_reward,
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

        return [
            add_mc_return(add_trajectory_reward(trajectory), gamma=self.gamma)
            for trajectory in trajectories
        ]

    def reset(self):
        """reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def update_env(self, env):
        """
        update the environment. new agents are created instead of reusing old ones
        """
        self.n_parallel_agents = env.batch_size

        self.agents = [
            self.agent_class(
                rollout_engine=self.rollout_engine,
                engine_name=self.engine_name,
                tokenizer=self.tokenizer,
                api_key=self.api_key,
                api_retries=self.api_retries,
                **self.kwargs,
            )
            for _ in range(self.n_parallel_agents)
        ]

        self.env = env