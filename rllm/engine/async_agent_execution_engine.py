from rllm.engine.agent_execution_engine import AgentExecutionEngine
import asyncio
from rllm.misc import colorful_print
from rllm.environments.env_utils import (
    add_trajectory_reward,
    add_mc_return,
    add_training_reward,
    compute_training_score,
    compute_environment_score,
)
from rllm.router.router import Router
import torch
import uuid

import openai


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
        max_episodes=5,
        max_trajectory_length=8192,
        max_prompt_length=1024,
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
        self.max_episodes = max_episodes
        self.agent_args = agent_args
        self.max_trajectory_length = max_trajectory_length
        self.max_prompt_length = max_prompt_length

        # self.agents = [agent_class(**agent_args) for _ in range(self.n_parallel_agents)]
        # self.envs = [env_class(**env_args) for _ in range(self.n_parallel_agents)]

        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", None)

        if engine_name == "openai":
            from openai import AsyncOpenAI 
            self.client = AsyncOpenAI(**self.rollout_engine_args)
            
        if self.engine_name == "verl":
            # All generation is done via scheduler. Currently only works for verl
            self.router = Router(rollout_engine=rollout_engine)

    # multithread safe generator function
    async def get_action_async(self, trajectory, agent, application_id, **kwargs):
        # currently load balancing only happens with verl
        if self.engine_name == "openai":
            return await self._get_action_openai_async(trajectory, agent, **kwargs)
        elif self.engine_name == "verl":
            return await self._get_action_verl_async(
                trajectory, agent, application_id, **kwargs
            )
        else:
            raise NotImplementedError

    async def _get_action_verl_async(self, trajectory, agent, application_id, **kwargs):
        """
        Asynchronous version for getting a single action from verl using Ray worker groups.
        """
        prompt = agent._pre_get_action(trajectory)
        batch = self._convert_prompt_verl([prompt], **kwargs)

        output = await self.router._get_result_verl_async(
            batch, application_id, **kwargs
        )

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
                    print("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    return f"Error processing content: {e}"

        response = await get_response(prompt)
        print("oai response:", response)
        return agent._post_get_action(response), response

    async def run_agent_trajectory(
        self, idx, application_id, seed=0, mode="Text", **kwargs
    ):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]

        # Reset environment with the task
        observation, _ = env.reset()

        # Reset agent
        agent.reset()

        # Initialize trajectory for this task
        trajectory = []

        # For verl training
        prompt_token_len = 0
        prompt_tokens = []
        response_len = 0
        response_tokens = []
        response_masks = []

        # For returning conversations
        conversations = []

        initial_messages = agent.format_observation_as_messages(observation)
        prompt_tokens, _ = self._convert_messages_to_tokens_and_masks(initial_messages)
        prompt_token_len = len(prompt_tokens)

        # Update conversation version
        conversations.extend(initial_messages)

        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length already exceeded max_prompt_length, retrying"
            )

        termination_reason = "episode_len"

        trajectory.append(
            {
                "next_observation": observation,
            }
        )
        for _ in range(self.max_episodes):
            # Get action from agent
            action, response = await self.get_action_async(
                trajectory, agent, application_id, **kwargs
            )

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
                info=info,
            )
            # Process response tokens
            assistant_msg = {"role": "assistant", "content": response}
            env_messages = agent.format_observation_as_messages(next_observation)
            
            assistant_msg_tokens, assistant_msg_masks = self._convert_message_to_tokens_and_masks(assistant_msg)
            env_msg_tokens, env_msg_masks = self._convert_messages_to_tokens_and_masks(env_messages)

            # Reached maximum number of tokens for the trajectory
            if (
                response_len
                + len(assistant_msg_tokens)
                + len(env_msg_tokens)
                + self.max_prompt_length
                >= self.max_trajectory_length
            ):
                # Truncation length
                truncation_length = (
                    self.max_trajectory_length - self.max_prompt_length - response_len
                )
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
                
                # Update conversation history
                conversations.append(assistant_msg)
                conversations.extend(env_messages)

                # Update trajectory (Though it is truncated)
                trajectory.append(
                    {
                        "observation": observation,
                        "next_observation": next_observation,
                        "reward": 0,  # TODO: May need to update this to be minimum environment score in the future
                        "done": terminated or truncated,
                        "action": action,
                        "info": info,
                        "response": response,
                        "truncated": True,
                    }
                )

                colorful_print(
                    f"Trajectory {idx} completed due to maximum trajectory length reached. But entire Text or Conversation will be returned. Reward is 0. \n",
                    "yellow",
                )
                termination_reason = ""  # no longer need, already logged
                # handle returning
                break
            
            # Update repsonse token length
            response_len += len(assistant_msg_tokens) + len(env_msg_tokens)

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens + env_msg_tokens)
            response_masks.extend(assistant_msg_masks + env_msg_masks)

            # Update conversation version
            conversations.append(assistant_msg)
            conversations.extend(env_messages)

            # Update the trajectory
            trajectory.append(
                {
                    "observation": observation,
                    "next_observation": next_observation,
                    "reward": reward,
                    "done": terminated or truncated,
                    "action": action,
                    "info": info,
                    "response": response,
                    "truncated": True,
                }
            )

            observation = next_observation

            # Check if episode is done
            if terminated or truncated:
                termination_reason = "termination" if terminated else "truncation"
                break

        if termination_reason:
            colorful_print(
                f"Trajectory {idx} completed due to {termination_reason}. Reward is {reward}. \n",
                "green",
            )

        trajectory = trajectory[1:]
        augmented_trajectory = add_mc_return(
            add_trajectory_reward(trajectory), gamma=self.gamma
        )
        training_reward = agent.compute_training_reward(augmented_trajectory)
        result_trajectory = add_training_reward(augmented_trajectory, training_reward)
        if mode == "Text":
            return result_trajectory

        if mode == "Token":
            # Collect into dictionary form
            training_reward = compute_training_score(result_trajectory)
            env_reward = compute_environment_score(result_trajectory)

            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "training_reward": training_reward,
                "environment_reward": env_reward,
                "uid": env.env_id,
            }
            return token_result

        if mode == "Conversation":
            return conversations

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

    async def interact_environment_generator(
        self, reset_seed=0, timing_raw={}, mode="Text", **kwargs
    ):
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
        i = 1
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                print(f"yielded {i}/{len(tasks)}")
                i += 1
                yield result
            except Exception as e:
                raise e
            
        if self.engine_name == "verl":
            self.router.__exit__()
