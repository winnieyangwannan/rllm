from rllm.models.batch_agent import BatchAgent
import asyncio
from verl.trainer.ppo.ray_trainer import _timer
from rllm.misc import colorful_print
from rllm.environments.env_utils import add_trajectory_reward, add_mc_return, add_training_reward, compute_training_score, compute_environment_score

import torch
import numpy as np
import time


class AgentExecutionEngine(BatchAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.envs = [self.env for _ in range(self.n_parallel_agents)]

    async def get_action_async(self, trajectory, agent, **kwargs):
        if self.engine_name == "openai":
            return await self._get_action_openai_async(trajectory, agent, **kwargs)
        elif self.engine_name == "verl":
            return await self._get_action_verl_async(trajectory, agent, **kwargs)
        else:
            raise NotImplementedError

    async def _get_action_verl_async(self, trajectory, agent, **kwargs):
        """
        Asynchronous version for getting a single action from verl.
        """
        from verl.protocol import pad_dataproto_to_divisor
        import asyncio

        prompt = agent._pre_get_action(trajectory)
        batch = self._convert_prompt_verl([prompt], **kwargs)
        
        # Pad to match worker group world size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.rollout_engine.world_size)
        
        # Add rollout_generator_id (-1 for padding)
        rollout_generator_id = np.array([0] + [-1] * pad_size, dtype=object)
        batch_padded.non_tensor_batch["rollout_generator_id"] = rollout_generator_id
        
        # Run the generator in a separate thread
        result = None
        
        def collect_result():
            nonlocal result
            gen_seq_generator = self.rollout_engine.generate_sequences_async(prompts=batch)
            for output in gen_seq_generator:
                idx = output.non_tensor_batch["rollout_generator_id"][0]
                if idx != -1:  # Skip padding
                    output_text = self.tokenizer.batch_decode(
                        output.batch["responses"], skip_special_tokens=False
                    )[0]  # Only one response
                    pad_token = self.tokenizer.pad_token
                    response = output_text.replace(pad_token, "")
                    result = agent._post_get_action(response)
                    break  # We only need the first result
        
        # Run collection in a separate thread
        await asyncio.to_thread(collect_result)
        
        return result

    async def _get_action_openai_async(self, trajectory, agent, **kwargs):
        import openai
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        prompt = agent._pre_get_action(trajectory)
        
        async def get_response(prompt):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await client.chat.completions.create(
                        messages=prompt,
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


    async def interact_environment_async(self):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.
        Returns a list of trajectories, one for each parallel agent.
        """
        # Initialize storage for trajectories
        trajectories = [[] for _ in range(self.n_parallel_agents)]
        
        observations = [env.reset()[0] for env in self.envs]
        for agent in self.agents:
            agent.reset()

        async def run_agent_episode(agent_idx):
            """Run a single agent's episode asynchronously"""
            agent = self.agents[agent_idx]
            env = self.envs[agent_idx]
            observation = observations[agent_idx]
            
            for _ in range(self.episode_len):
                # Get action from agent
                action = await self.get_action_async(observation, agent, **agent.sampling_params)
                
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
                trajectories[agent_idx].extend(agent.trajectory)
                
                observation = next_observation
                
                # Check if episode is done
                if terminated or truncated:
                    break
        
        # Create tasks for all agents to run independently
        tasks = [
            asyncio.create_task(run_agent_episode(agent_idx)) 
            for agent_idx in range(self.n_parallel_agents)
        ]
        
        # Wait for all episodes to complete
        await asyncio.gather(*tasks)
                
        return trajectories
    

    async def run(self, reset_seed=0, timing_raw={}, mode="Text", **kwargs):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.
        
        Args:
            reset_seed (int, optional): The random seed for resetting the environment. Defaults to 0.
            timing_raw (dict, optional): Dictionary for tracking execution times of different stages. Defaults to {}.
            mode (str, optional): Decides the return value. If `Text`,returns structured text. If 'Token', returns tokenized responses. If 'Conversation', returns the conversation format. Defaults to Text.
            **kwargs: Additional arguments that may be passed to environment interactions.
            
        Returns:
            Same as interact_environment, depending on the mode parameter.
        """
        assert mode in ["Text", "Token", "Conversation"], f"Return mode {mode} not supported"
        
        # Initialize storage for trajectories and token data
        trajectories = [[] for _ in range(self.n_parallel_agents)]
        all_prompt_tokens = [[] for _ in range(self.n_parallel_agents)]
        all_response_tokens = [[] for _ in range(self.n_parallel_agents)]
        all_response_masks = [[] for _ in range(self.n_parallel_agents)]
        all_conversations = [[] for _ in range(self.n_parallel_agents)]
        
        # Reset environments and agents
        observations = [env.reset(seed=reset_seed)[0] for env in self.envs]
        for agent in self.agents:
            agent.reset()
            
        # Process initial observations
        max_prompt_token_len = 0
        for i, obs in enumerate(observations):
            # Add initial observation to trajectories
            trajectories[i].append({"next_observation": obs})
            
            # Process initial prompt tokens
            initial_msg = {
                "role": "user",
                "content": self.agents[i].convert_observation_to_string(obs, with_system_prompt=True),
            }
            prompt_tokens, _ = self._convert_message_to_tokens_and_masks(initial_msg)
            max_prompt_token_len = max(max_prompt_token_len, len(prompt_tokens))
            all_prompt_tokens[i] = prompt_tokens
            
            # Add to conversations
            all_conversations[i].append(initial_msg)
        
        if max_prompt_token_len >= self.max_trajectory_length:
            self.reset()
            raise Exception("Initial prompt length already exceeded max_trajectory_length")

        async def run_agent_episode(agent_idx):
            """Run a single agent's episode asynchronously"""
            agent = self.agents[agent_idx]
            env = self.envs[agent_idx]
            observation = observations[agent_idx]
            response_token_len = 0
            
            for step in range(self.episode_len):
                # Get action from agent
                with _timer("get_action", timing_raw):
                    action = await self.get_action_async(observation, agent, **agent.sampling_params)
                    response = action  # In this implementation, action is the response
                
                # Take step in environment
                with _timer("env_step", timing_raw):
                    next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Process tokens for this step
                assistant_msg = {"role": "assistant", "content": response}
                next_obs_txt = agent.convert_observation_to_string(next_observation, with_system_prompt=False)
                env_msg = {"role": "user", "content": next_obs_txt}
                
                assistant_msg_tokens, assistant_msg_masks = self._convert_message_to_tokens_and_masks(assistant_msg)
                env_msg_tokens, env_msg_masks = self._convert_message_to_tokens_and_masks(env_msg)
                
                # Check if we'll exceed token limit
                is_truncated = False
                if response_token_len + len(assistant_msg_tokens) + len(env_msg_tokens) + max_prompt_token_len >= self.max_trajectory_length:
                    # Calculate truncation length
                    truncation_length = self.max_trajectory_length - max_prompt_token_len - response_token_len
                    # Truncate tokens and masks
                    combined_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    combined_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                    is_truncated = True
                else:
                    combined_tokens = assistant_msg_tokens + env_msg_tokens
                    combined_masks = assistant_msg_masks + env_msg_masks
                
                # Update token collections
                all_response_tokens[agent_idx].extend(combined_tokens)
                all_response_masks[agent_idx].extend(combined_masks)
                response_token_len += len(combined_tokens)
                
                # Update conversations
                all_conversations[agent_idx].append(assistant_msg)
                all_conversations[agent_idx].append(env_msg)
                
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
                
                # Add step to trajectory
                trajectories[agent_idx].append({
                    "observation": observation,
                    "next_observation": next_observation,
                    "reward": reward,
                    "done": terminated or truncated,
                    "action": action,
                    "info": info,
                    "response": response,
                    "truncated": is_truncated
                })
                
                observation = next_observation
                
                # Check if episode is done
                if terminated or truncated or is_truncated:
                    break
        
        # Create tasks for all agents to run independently
        tasks = [
            asyncio.create_task(run_agent_episode(agent_idx)) 
            for agent_idx in range(self.n_parallel_agents)
        ]
        
        # Wait for all episodes to complete
        await asyncio.gather(*tasks)
        
        # Remove sentinel and process trajectories
        trajectories = [traj[1:] if traj else [] for traj in trajectories]
        trajectory_result = []
        
        for i, trajectory in enumerate(trajectories):
            augmented_trajectory = add_mc_return(add_trajectory_reward(trajectory), gamma=self.gamma)
            training_reward = self.agents[i].compute_training_reward(augmented_trajectory)
            trajectory_result.append(add_training_reward(augmented_trajectory, training_reward))
        
        # Return based on mode
        if mode == "Text":
            return trajectory_result
        
        if mode == "Token":
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