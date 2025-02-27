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

    def get_actions_yield(self, observations, obs_idxs=[], **kwargs):
        """
        Yield actions with same size as the observations list, may be out of order. 
        obs_idxs: The list of indexes that the observations are from. Used to index into self.agents

        Yield: the tuple (action, obs_idx it corresponds to, response)
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
        rollout_generator_id = np.array(obs_idxs + [-1] * pad_size, dtype=object)
        batch_padded.non_tensor_batch["rollout_generator_id"] = rollout_generator_id
        gen_seq_generator = self.rollout_engine.generate_sequences_async(prompts=batch_padded)
        for output in gen_seq_generator:
            idx = output.non_tensor_batch["rollout_generator_id"][0]
            if idx != -1:
                output_text = self.tokenizer.batch_decode(
                    output.batch["responses"], skip_special_tokens=False
                )
                assert len(output_text) == 1, "Only 1 action should be yielded at one time"
                for i, text in enumerate(output_text):
                    pad_token = self.tokenizer.pad_token
                    response = text.replace(pad_token, "")
                    action = self.agents[idx]._post_get_action(response)
                    yield action, idx, response


    def interact_environment_generator(self, reset_seed=0, timing_raw={}, **kwargs):
        """
        Runs the trajectory collection loop and yields completed trajectories as they finish.
        Uses a simple queue-based async mechanism to process available results.
        """
        assert self.env, f"Env cannot be empty, but got {self.env}"
        assert hasattr(self.env, "batch_size"), "Env does not have batch_size attribute"

        env_batch_size = self.env.batch_size
        assert (
            env_batch_size == self.n_parallel_agents
        ), "Number of parallel environments should match number of parallel agents."

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        trajectories = [[] for _ in range(env_batch_size)]
        steps = [0 for _ in range(env_batch_size)]
        observations, infos = self.env.reset(seed=reset_seed)
        batch_done = [False for _ in range(env_batch_size)]
        responses_queue = asyncio.Queue()
        results_queue = asyncio.Queue() 

        pending_obs = []
        pending_traj_idxs = []
        last_submit_time = time.time()

        batch_size = self.rollout_engine.world_size
        timeout_seconds = 3

        async def async_action_worker(obs_batch, traj_idxs):
            """Runs `get_actions_yield()` asynchronously in a thread."""
            for action, traj_idx, response in await asyncio.to_thread(self.get_actions_yield, obs_batch, traj_idxs, **kwargs):
                await responses_queue.put((action, traj_idx, response))

        async def flush_pending_actions():
            """Flush accumulated actions when conditions are met."""
            nonlocal pending_obs, pending_traj_idxs, last_submit_time
            if pending_obs:
                await async_action_worker(pending_obs, pending_traj_idxs)
                pending_obs = []
                pending_traj_idxs = []
                last_submit_time = time.time()

        async def async_loop():
            """Main async loop handling actions and stepping through the environment."""
            # Start processing initial batch
            loop.create_task(async_action_worker(observations, list(range(env_batch_size))))

            while not all(batch_done):
                traj_idx_list = []
                actions_list = []
                responses_list = []

                # Process available actions without blocking
                while not responses_queue.empty():
                    action, traj_idx, response = await responses_queue.get()
                    traj_idx_list.append(traj_idx)
                    actions_list.append(action)
                    responses_list.append(response)

                if not actions_list:
                    if time.time() - last_submit_time > timeout_seconds:
                        await flush_pending_actions()
                    await asyncio.sleep(0.01)  # Yield control to prevent blocking
                    continue

                

                try:
                    next_observations, rewards, terminateds, truncateds, infos = (
                        self.env.step(actions=actions_list, env_idxs=traj_idx_list)
                    )
                except Exception as e:
                    print(e)
                    self.reset()
                    raise (e)

                for i, traj_idx in enumerate(traj_idx_list):

                    # TODO: make sure it works
                    aug_reward= self.agents[traj_idx].augment_reward(responses_list[i], next_observations[i], rewards[i])

                    trajectories[traj_idx].append({
                        "observation": observations[traj_idx],
                        "next_observation": next_observations[i],
                        "reward": rewards[i],
                        "done": terminateds[i] or truncateds[i],
                        "action": actions_list[i],
                        "info": infos[i],
                        "augmented_reward": aug_reward,
                        "response": responses_list[i],
                    })

                    self.agents[traj_idx].update(
                        actions_list[i],
                        observations[traj_idx],
                        next_observations[i],
                        rewards[i],
                        terminateds[i],
                        truncateds[i],
                        infos[i],
                    )

                    observations[traj_idx] = next_observations[i]
                    steps[traj_idx] += 1

                    if steps[traj_idx] > self.episode_len or terminateds[i] or truncateds[i]:
                        termination_reason = 'termination' if terminateds[i] else 'truncation'
                        if steps[traj_idx] > self.episode_len:
                            termination_reason = "episode_len"
                        colorful_print(
                            f"Trajectory {traj_idx} completed due to {termination_reason}. Reward is {rewards[i]}. \n",
                            "green",
                        )
                        batch_done[traj_idx] = True
                        result = add_mc_return(
                            add_trajectory_reward(trajectory=trajectories[traj_idx]),
                            gamma=self.gamma,
                        ), traj_idx
                        await results_queue.put(result) 
                    else:
                        pending_obs.append(next_observations[i])
                        pending_traj_idxs.append(traj_idx)

                        if len(pending_obs) >= batch_size:
                            await flush_pending_actions()

            await results_queue.put(None) 

        loop.create_task(async_loop())  

        try:
            while True:
                result = loop.run_until_complete(results_queue.get())
                if result is None:
                    break  
                yield result

        finally:
            # Clean up
            pending = asyncio.all_tasks(loop)
            for task in pending:
                print("One trajectory generation task is cancelled")
                task.cancel()
            
            if not loop.is_closed():
                # Run loop one final time to execute any remaining callbacks
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        

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