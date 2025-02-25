import openai
import time
import os
from tqdm import tqdm
import multiprocessing as mp
import torch
import numpy as np
import accelerate
import queue
import threading
import asyncio

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
        return actions

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
        return actions

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
        return actions

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
        for i, done in enumerate(batch_done):
            if not done and observations[i] is not None:
                new_observations.append(observations[i])
                obs_idxs.append(i)

        if len(new_observations) > 0:
            outputs = []
            for i in range(0, len(new_observations), self.safe_batch_size):
                try:
                    outputs += self.get_actions(
                        new_observations[i : i + self.safe_batch_size],
                        obs_idxs[i : i + self.safe_batch_size],
                        **kwargs,
                    )
                except Exception as e:
                    # sometime it says unable to infer image channel error
                    colorful_print(f"Error in getting action: {e}", "red")
                    # outputs += ["ERROR"]*len(new_observations[i:i+safe_batch_size])
                    raise e

            for i, idx in enumerate(obs_idxs):
                actions[idx] = outputs[i]

        for action, obs in zip(actions, observations):
            if obs is None:
                assert action == "", "Action should be empty, First assert"

        return actions

    def interact_environment(self, reset_seed=0, timing_raw={}, **kwargs):
        """
        Run environment interactions in batches.
        Collects trajectories in the format:
        [[{"observation":, "next_observation":, "reward":, "done":, "action": },...],...]
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
                        actions = self._safe_get_actions(
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

    def get_actions_yield(self, observations, obs_idxs=[], **kwargs):
        """
        Yield actions with same size as the observations list, may be out of order. Yielding the tuple (response, obs_idx it corresponds to)
        obs_idxs: The list of indexes that the observations are from. Used to index into self.agents
        """
        if obs_idxs:
            assert len(observations) == len(
                obs_idxs
            ), f"Number of observations {len(observations)} should equal to the number of agents they are for ({len(obs_idxs)})"
        assert (
            self.engine_name == "verl"
        ), "Currently only veRL is supported for trajectory yielding"
        yield from self._get_actions_verl_yield(observations, obs_idxs, **kwargs)

    def _get_actions_verl_yield(self, observations, obs_idxs=[], **kwargs):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        prompts = [
            self.agents[obs_idxs[i]]._pre_get_action(obs)
            for i, obs in enumerate(observations)
        ]

        batch = self._convert_prompt_verl(prompts, **kwargs)
        # because of veRL's chunking. we need to pad number of prompts to be a multiple of worker group world size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.rollout_engine.world_size)
        
        # augment the data with non_tensor_batch "rollout_generator_id" to tell which agent/trajectory it belongs to. -1 mean padding
        rollout_generator_id = np.array(obs_idxs + [-1] * pad_size)
        batch_padded.non_tensor_batch["rollout_generator_id"] = rollout_generator_id

        gen_seq_generator = self.rollout_engine.generate_sequences_async(prompts=batch_padded)
        for output in gen_seq_generator:
            # TODO: check if this is correct to do output.non_tensor_batch['rollout_generator_id'] for int
            idx = output.non_tensor_batch["rollout_generator_id"]
            if idx != -1:
                output_text = self.tokenizer.batch_decode(
                    output.batch["responses"], skip_special_tokens=False
                )
                pad_token = self.tokenizer.pad_token
                response = output_text.replace(pad_token, "")
                action = self.agents[idx]._post_get_action(response)
                yield action, idx

    def interact_environment_generator(self, reset_seed=0, timing_raw={}, **kwargs):
        """
        Same as interact_environment but async and yielding trajectories as they terminate. May be out of order.
        """
        assert self.env, f"Env cannot be empty, but got {self.env}"
        assert hasattr(self.env, "batch_size"), "Env does not have batch_size attribute"
        env_batch_size = self.env.batch_size
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # This is veRL specific to avoid padding
        world_size = self.rollout_engine.world_size
        timeout = 0.5
        try:
            trajectories = [[] for _ in range(env_batch_size)]
            steps = [0 for _ in range(env_batch_size)]
            observations, infos = self.env.reset(seed=reset_seed)
            batch_done = [False for _ in range(env_batch_size)]

            # Queues for batch processing. Initially everything is pending
            pending_obs_queue = observations[:]
            pending_traj_idxs = [i for i in range(env_batch_size)]
            last_submit_time = time.time()
            responses_queue = asyncio.Queue()

            async def handle_action_request(obs_batch, traj_idxs):
                """Handles async action requests."""
                async for action, traj_idx in self.get_actions_yield(
                    obs_batch, obs_idxs=traj_idxs, **kwargs
                ):
                    await responses_queue.put((action, traj_idx))

            async def async_action_worker():
                """Continuously fetch actions from `get_actions_yield` and populate the queue."""
                while not all(batch_done):
                    if pending_obs_queue:
                        # Ensure batch size is a multiple of world_size or timeout occurs or number of trajectories left is not enough
                        if (
                            len(pending_obs_queue) % world_size == 0
                            or (time.time() - last_submit_time) > timeout
                            or sum(batch_done) < world_size
                        ):
                            current_obs = pending_obs_queue[:]
                            current_traj_idxs = pending_traj_idxs[:]
                            pending_obs_queue.clear()
                            pending_traj_idxs.clear()
                            last_submit_time = time.time()

                            loop.create_task(
                                handle_action_request(current_obs, current_traj_idxs)
                            )

                    await asyncio.sleep(0.01)  # Prevent CPU overuse

            loop.create_task(async_action_worker())

            while not all(batch_done):
                actions_map = {}

                while not responses_queue.empty():
                    action, traj_idx = responses_queue.get()
                    actions_map[traj_idx] = action

                if not actions_map:
                    os.sleep(0.01)  # Prevent busy waiting
                    continue

                traj_idx_list = list(actions_map.keys())
                actions = list(actions_map.values())

                try:
                    next_observations, rewards, terminateds, truncateds, infos = (
                        self.env.step(actions=actions, env_idxs=traj_idx_list)
                    )
                except Exception as e:
                    print(e)
                    self.reset()
                    raise (e)

                for i, traj_idx in enumerate(traj_idx_list):
                    trajectories[traj_idx].append(
                        {
                            "observation": observations[traj_idx],
                            "next_observation": next_observations[i],
                            "reward": rewards[i],
                            "done": terminateds[i] or truncateds[i],
                            "action": actions[i],
                            "info": infos[i],
                        }
                    )

                    self.agents[traj_idx].update(
                        actions[i],
                        observations[traj_idx],
                        next_observations[i],
                        rewards[i],
                        terminateds[i],
                        truncateds[i],
                        infos[i],
                    )

                    observations[traj_idx] = next_observations[i]

                    # If an environment is done, handle the completed trajectory
                    steps[traj_idx] += 1

                    if (
                        steps[traj_idx] > self.episode_len
                        or terminateds[i]
                        or truncateds[i]
                    ):
                        batch_done[i] = True
                        complete_reason = (
                            "termination" if terminateds[i] else "truncation"
                        )
                        if steps[traj_idx] > self.episode_len:
                            complete_reason = "reach step limit"
                        colorful_print(
                            f"Trajectory {traj_idx} completed due to {complete_reason}. Reward is {rewards[i]}. \n",
                            "green",
                        )
                        yield add_mc_return(
                            add_trajectory_reward(trajectory=trajectories[traj_idx]),
                            gamma=self.gamma,
                        )
                    else:
                        # put back into the pending lists
                        pending_obs_queue.append(next_observations[i])
                        pending_traj_idxs.append(traj_idx)

        except Exception as e:
            print(f"Error in environment interaction")
            import traceback

            print(traceback.format_exc())
            print(e)

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
