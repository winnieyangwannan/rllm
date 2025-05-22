import concurrent.futures
import asyncio
from threading import Thread
from queue import Queue
from typing import Type, Dict, List
import numpy as np
import torch
from pprint import pprint
from omegaconf import OmegaConf, open_dict
import uuid
from functools import partial
from jinja2 import Template
import re

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    WorkerType,
    ResourcePoolManager,
    RayWorkerGroup,
    compute_response_mask,
    compute_timing_metrics, 
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
    _timer,
)
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
import math
from functools import reduce
class AgentPPOTrainer(RayPPOTrainer):

    def __init__(
            self,
            config,
            tokenizer,
            role_worker_mapping: dict[Role, WorkerType],
            resource_pool_manager: ResourcePoolManager,
            ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            env_class=None,
            agent_class=None,
        ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping,
                         resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls,
                         reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class

        if self.config.agent.step_advantage_broadcast:
            print(f"Using step-wise advantage broadcasting, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print(f"Using trajectory advantage, max_prompt_length and max_response_length will be applied episode-wise")
    
    def init_workers(self):
        super().init_workers()

        # Initialize additional agent class 
        # Number of agents is set to be 0 initially
        if self.hybrid_engine: 
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg
        
        if self.config.agent.async_engine:
            if self.config.actor_rollout_ref.rollout.mode == "async":
                rollout_engine = self.async_rollout_manager
            else:
                rollout_engine = agent_rollout_wg

            self.agent_execution_engine = AsyncAgentExecutionEngine(
                rollout_engine=rollout_engine,
                config=self.config,
                engine_name="verl",
                tokenizer=self.tokenizer,
                model_path=self.config.actor_rollout_ref.model.path,
                max_steps=self.config.agent.max_steps,
                max_response_length=self.config.data.max_response_length,
                max_prompt_length=self.config.data.max_prompt_length,
                n_parallel_agents=self.config.agent.n_parallel_agents,
                agent_class=self.agent_class,
                agent_args=self.config.agent.get("agent_args", {}),
                env_class=self.env_class,
                env_args=self.config.env.get("env_args", {}),
                enforce_max_prompt_length=self.config.agent.step_advantage_broadcast,
                trajectory_timeout=self.config.agent.trajectory_timeout,
            )
        else:
            self.agent_execution_engine = AgentExecutionEngine(
                rollout_engine=agent_rollout_wg,
                engine_name="verl",
                tokenizer=self.tokenizer,
                config=self.config,
                model_path=self.config.actor_rollout_ref.model.path,
                max_steps=self.config.agent.max_steps,
                max_response_length=self.config.data.max_response_length,
                max_prompt_length=self.config.data.max_prompt_length,
                enforce_max_prompt_length=self.config.agent.step_advantage_broadcast,
            )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        envs = [self.env_class.from_json({**env_args[i], **self.config.env.get("env_args", {})}) for i in range(len(env_args))]
        agents = [self.agent_class(**self.config.agent.get("agent_args", {})) for _ in range(len(envs))]
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                with _timer("step", timing_raw):
                    self.init_envs_and_agents(batch)
                    
                    if not self.config.agent.step_advantage_broadcast:
                        final_gen_batch_output = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                    else:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info)
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.repeat_by_counts(repeat_counts, interleave=True)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None) # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._masked_pad_to_update_world_size(batch=batch)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:  
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"] # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(
                                -1
                            )  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.trainer.rejection_sample:
                            
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch)

                            if self.config.agent.step_advantage_broadcast:
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]  
                                batch = batch.select_idxs(non_pad_step_indices) # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step)[0]  
                                not_last_step_indices = np.where(is_last_step == False)[0]  
                                last_step_batch = batch.select_idxs(valid_last_step_indices) # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0] # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(
                                    last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool
                                )
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]
                                last_step_batch = dataprotoitem_to_dataproto(last_step_batch) # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]
                                non_last_step_batch = dataprotoitem_to_dataproto(non_last_step_batch)

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._masked_pad_to_update_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    batch.batch["input_ids"].shape[0]
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(
                                    batch.batch["input_ids"].shape[0], dtype=torch.bool
                                )
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]
                                batch = dataprotoitem_to_dataproto(batch)

                                

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]


                        if self.config.agent.step_advantage_broadcast:
                            # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                            is_last_step = batch.non_tensor_batch["is_last_step"]
                            last_step_indices = np.where(is_last_step)[0]  
                            other_step_indices = np.where(is_last_step == False)[0]  
                            other_step_batch = batch.select_idxs(other_step_indices)
                            batch = batch.select_idxs(last_step_indices) # This batch only has last steps
                            

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.agent.step_advantage_broadcast:
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        env_rewards_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True
            }
            self.init_envs_and_agents(test_batch)

            if not self.config.agent.step_advantage_broadcast:
                test_output_gen_batch = self.generate_agent_trajectory(
                    meta_info=test_batch.meta_info
                )
            else:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info)
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step)[0]  
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices) # This batch only has last steps
            
            test_batch = test_batch.union(test_output_gen_batch)

            # use environment score to report validation reward
            reward_tensor = test_batch.batch["token_level_scores"] 
            env_reward_tensor = test_batch.batch["environment_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            env_rewards_lst.append(env_reward_tensor.sum(-1).cpu())
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        env_reward_tensor = torch.cat(env_rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_env_reward = {}

        # to group for pass@k
        uid_tensor = test_batch.non_tensor_batch["uid"] 
        data_source_uid_pass_rates = {} # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            if data_source not in data_source_env_reward:
                data_source_env_reward[data_source] = []
            data_source_env_reward[data_source].append(env_reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0 # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

            
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        for data_source, env_rewards in data_source_env_reward.items():
            metric_dict[f"val/env_score/{data_source}"] = np.mean(env_rewards)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1) # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

    def generate_agent_trajectory(self, timing_raw={}, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
        """
        with _timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                trajectories = self.agent_execution_engine.generate_trajectories(
                    timing_raw=timing_raw, mode="Token", meta_info=meta_info
                )
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_trajectories(
                trajectories
            )
        return final_gen_batch_output

    def generate_agent_steps(self, timing_raw={}, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        with _timer("collect_trajectory", timing_raw):
            steps = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                steps = self.agent_execution_engine.generate_trajectories(
                    timing_raw=timing_raw, mode="Step", meta_info=meta_info
                )
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(
                steps
            )
        return final_gen_batch_output


    def _transform_agent_trajectories(self, trajectories: List[Dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length
        
        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        environment_scores = []

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["training_reward"])
            environment_scores.append(traj["environment_reward"])

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list], 
            batch_first=True,  
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])        

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)                   

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)           

        traj_mask = torch.nn.utils.rnn.pad_sequence(
            all_masks_list, batch_first=True, padding_value=0
        )
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False) 

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        environment_score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
                environment_score_batch[i, last_valid_idx] = environment_scores[i]

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
            "environment_scores": environment_score_batch,
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch)
    


    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.
        
        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]
        environment_scores = tensor_batch.batch["environment_scores"]
        
        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)
        
        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)
            
            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)
            
            colorful_print("Prompt:", fg="green", bold=True)
            colorful_print(f"{prompt_text}\n", fg="green")
            
            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]
            
            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]
            
            # Then show token-by-token with masking
            colorful_print("Response with masking:", fg="yellow", bold=True)
            
            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask)):
                token_text = self.tokenizer.decode(token)
                
                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0
                has_env_reward = environment_scores[i, j] != 0
                
                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward or has_env_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")
                    
                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"
                    if has_env_reward:
                        reward_info += f" ER:{environment_scores[i, j].item():.2f}"
                    
                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")
            
            print()  # New line after all tokens
            
            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            total_env_reward = environment_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Training={total_reward:.2f}, Environment={total_env_reward:.2f}")


    def generate_agent_trajectories_async(self, timing_raw={}, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        queue = Queue()
        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done
            asyncio.run(consume())
        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: List[Dict]):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = [] 
        all_responses_list = []

        step_numbers = [] # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = [] # total number of steps the trajectory this step belongs to have
        training_rewards = []
        environment_rewards = []
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode['training_reward']
            environment_reward = episode['environment_reward']

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            environment_rewards.append(environment_reward)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])

        
        # Convert all steps into token tensors
        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list], 
            batch_first=True,  
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])        

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)                   

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)    

        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(complete_step_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # same as regular repsonse_mask, padded tensors will have this zeroed out
        traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0) 

        # Place all rewards to last response token of the last_step response
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        environment_score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)
        
        # reward is given for last token of every step for logging purposes, but only last steps will be used to calculate advantage
        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1 # since step_numbers is 0 indexed
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                    score_batch[step_index, last_valid_idx] = traj_score
                    environment_score_batch[step_index, last_valid_idx] = environment_rewards[i]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "environment_scores": environment_score_batch,
            "traj_mask": traj_mask, 
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list, dtype=int),
            "step_nums": np.array(all_steps_step_num, dtype=int),
            "is_last_step": np.array(all_steps_is_last_step_list, dtype=bool),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))], dtype=bool),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))], dtype=str), # in case need to differentiate which iteration the step is coming from
        }

        meta_info = {
            "repeat_counts": [x + 1 for x in step_numbers]
        }

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        self.visualize_trajectory(result, sample_idx=0, max_samples=2)
        return result

       
    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch['response_mask'] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch['response_mask'] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]   
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]             
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]                
        src_mask = last_step_batch.batch["response_mask"]             
        tgt_mask = other_step_batch.batch["response_mask"]           

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean() 

            if self.config.agent.normalize_step_advantage:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([
            torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32)
            for i, idx in enumerate(tgt_indices)
        ])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

        # Assign 0 advantage to those 
        is_pad = torch.from_numpy(other_step_batch.non_tensor_batch["is_pad_step"]).bool()
        other_step_batch.batch["advantages"][is_pad] = 0.0
        other_step_batch.batch["returns"][is_pad] = 0.0

    def _masked_pad_to_update_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch
        
        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
        
        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch