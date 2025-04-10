import uuid
from pprint import pprint
import torch
from copy import deepcopy
import time
import threading
from threading import Thread
from queue import Queue
import asyncio
import numpy as np

from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine

from rllm.trainer.agent_trainer import AgentPPOTrainer 

from verl.trainer.ppo.ray_trainer_pipeline import (
    Timer,
    update_metrics,
    SortedQueue,
)

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    WorkerType,
    ResourcePoolManager,
    RayWorkerGroup,
    compute_timing_metrics, 
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
    _timer,
)
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs



class AsyncAgentPPOTrainer(AgentPPOTrainer):

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        assert Role.Actor in self.role_worker_mapping and Role.Rollout in self.role_worker_mapping, "Actor and Rollout must be in role_worker_mapping"
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        # actor_gpu_ids = actor_resource_pool.gpu_assignments if isinstance(actor_resource_pool, RayResourcePool) else None

        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role='actor',
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[actor_resource_pool]['actor'] = actor_cls

        # Get rollout resource pool
        rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        # rollout_gpu_ids = rollout_resource_pool.gpu_assignments if isinstance(rollout_resource_pool, RayResourcePool) else None
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role='rollout',
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[rollout_resource_pool]['rollout'] = rollout_cls

        self.actor_wg = RayWorkerGroup(resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls)
        self.rollout_wg = RayWorkerGroup(resource_pool=rollout_resource_pool, ray_cls_with_init=rollout_cls)

        self.actor_wg.init_model()
        self.rollout_wg.init_model()

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.rollout_wg,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            episode_len=self.config.agent.trajectory_episode_len,
            max_trajectory_length=self.config.agent.max_trajectory_length,
            max_prompt_length=self.config.data.max_prompt_length,
        )

    def init_envs(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        envs = [self.env_class(**env_args[i]) for i in range(len(env_args))]
        batch.non_tensor_batch["uid"] = np.array([env.env_id for env in envs], dtype=object)

        return envs
    
    def fit_agent(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate_agent()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        replay_queue = Queue()
        total_mini_batch_iters = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_iter, batch_dict in enumerate(self.train_dataloader):
                print(f"step: {self.global_steps}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                # must pop those keys for generation so they no longer exist
                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                metrics = {}
                timing_raw = {}

                envs = self.init_envs(batch)
                agents = [self.agent_class() for _ in range(len(envs))]
                
                with Timer('step', timing_raw):

                    def create_replay_queue(generator, q):
                        uid_to_trajectories = {} # mapping of environment id (uid) to trajectories. Only put to queue in groups of size self.config.actor_rollout_ref.rollout.n
                        with Timer('gen', timing_raw):
                            for _, trajectory in enumerate(generator):
                                uid = trajectory['uid']
                                if uid not in uid_to_trajectories:
                                    uid_to_trajectories[uid] = []
                                uid_to_trajectories[uid].append(trajectory)
                                if len(uid_to_trajectories[uid]) == self.config.actor_rollout_ref.rollout.n:
                                    q.put((batch_iter, uid_to_trajectories[uid]))    
                                    del uid_to_trajectories[uid] # so even if there is replicas it's still grouped correctly

                    # Get the generator function which will yield results as they complete
                    gen_seq_generator = self.generate_agent_trajectories_async(envs, agents, timing_raw=timing_raw, meta_info=batch.meta_info)
                    thread = threading.Thread(target=create_replay_queue, args=(gen_seq_generator, replay_queue))
                    thread.start()
                    
                    ppo_train_batch_size =  self.config.data.train_batch_size
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    assert ppo_train_batch_size % ppo_mini_batch_size == 0, "PPO mini batch size must be a divisor of the total training batch size"
                    ppo_step_minibatch_iter = ppo_train_batch_size // ppo_mini_batch_size
                    num_loops = ppo_step_minibatch_iter #ppo_step_minibatch_iter +1 if batch_iter > 0 else  ppo_step_minibatch_iter 
                    # Initialize Empty data proto
                    training_batch = []
                    for mini_batch_iter in range(num_loops):
                        print(f"mini_batch_iter: {mini_batch_iter + 1} / {num_loops}", flush=True)
                        if mini_batch_iter == num_loops - 1:
                            while True:
                                if replay_queue.qsize() >= ppo_mini_batch_size:
                                    break
                                print("waiting for last item")
                                time.sleep(1)
                            break
                        mini_batch_metrics = {}
                        start_time = time.perf_counter()         
                        with Timer('pipeline_gen', timing_raw):
                            trajectories = []
                            for _ in range(ppo_mini_batch_size):
                                _, trajes = replay_queue.get()
                                trajectories.extend(trajes)
                            mini_batch = self._transform_agent_trajectories(trajectories=trajectories)
                            mini_batch.non_tensor_batch["uid"] = np.array([traj["uid"] for traj in trajectories], dtype=object)
                        end_time = time.perf_counter()
                        print(f"Generate mini batch took {end_time - start_time:.2f} seconds")
                        
                        if total_mini_batch_iters % ppo_step_minibatch_iter == ppo_step_minibatch_iter - 1:
                            mini_batch.meta_info['last_mini_batch'] = True

                        with Timer('adv', timing_raw):
                            reward_tensor = mini_batch.batch['token_level_scores'] # already computed
                            print('Reward tensor:', reward_tensor.sum(-1))
                        
                            # Rejection sampling based on rewards
                            # Group rewards by uid
                            uids = mini_batch.non_tensor_batch['uid']
                            unique_uids = np.unique(uids)
                            valid_mask = torch.ones(len(uids), dtype=torch.bool)
                            solve_none = 0
                            solve_all = 0
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                                
                                # Check if all rewards are 0 or all are 1 for this uid
                                if (uid_rewards == 0).all():
                                    valid_mask[uid_mask] = False
                                    solve_none += 1
                                elif (uid_rewards == 1).all():
                                    valid_mask[uid_mask] = False
                                    solve_all += 1
                            
                            # Log to metrics
                            mini_batch_metrics['batch/solve_none'] = solve_none
                            mini_batch_metrics['batch/solve_all'] = solve_all
                            
                            
                            if self.config.actor_rollout_ref.rollout.vllm_log_prob:
                                # Avoid recompute log_prob bugs. Log probs from vLLM. (Could be buggy)
                                mini_batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                                mini_batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                                mini_batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                                mini_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                            else:
                                # Recompute old_log_probs using Pytorch FSDP.
                                with Timer('old_log_prob', timing_raw):
                                    old_log_prob = self.actor_wg.compute_log_prob(mini_batch)
                                    mini_batch = mini_batch.union(old_log_prob)

                            if self.use_reference_policy:
                                # compute reference log_prob
                                with Timer('ref', timing_raw):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(mini_batch)
                                    mini_batch = mini_batch.union(ref_log_prob)
                            
                            mini_batch.batch['token_level_rewards'] = mini_batch.batch['token_level_scores']
                                                    # compute advantages, executed on the driver process
                            mini_batch = compute_advantage(mini_batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam)

                        self._balance_batch(mini_batch, metrics=mini_batch_metrics)
                        # compute global_valid tokens
                        mini_batch.meta_info['global_token_num'] = torch.sum(mini_batch.batch['attention_mask'], dim=-1).tolist()
                        # update actor
                        start_time = time.perf_counter()
                        with Timer('update_actor', timing_raw):
                            actor_output = self.actor_wg.update_actor_mini_batch(mini_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        end_time = time.perf_counter()
                        print(f"Actor update took {end_time - start_time:.2f} seconds", flush=True)
                        mini_batch_metrics.update(actor_output_metrics)
                        training_batch.append(mini_batch)
                        update_metrics(metrics, mini_batch_metrics)
                        total_mini_batch_iters += 1

                    # last_iter_mini_batch_iter = (mini_batch_iter + last_iter_mini_batch_iter - 1) % ppo_step_minibatch_iter
                    with Timer('rollout_model_update', timing_raw):
                        updated_actor_module_fsdp_ref = self.actor_wg.get_state_dict()
                        if isinstance(updated_actor_module_fsdp_ref, list):
                            updated_actor_module_fsdp_ref = updated_actor_module_fsdp_ref[0]
                        self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp_ref)
                    training_batch = DataProto.concat(training_batch)
                    
                    # Validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with Timer('testing', timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()    
                
                metrics.update(compute_data_metrics(batch=training_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=training_batch, timing_raw=timing_raw))
                
                
                for k, v in metrics.items():
                    if isinstance(v, (list, np.ndarray)):
                        if 'batch/' in k:
                            metrics[k] = np.sum(v)
                        else:
                            metrics[k] = np.mean(v)
                
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if self.val_reward_fn is not None and self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    val_metrics = self._validate_agent()
                    pprint(f'Final validation metrics: {val_metrics}')
                    logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return


    def _validate_agent(self):
        rewards_lst = []
        env_rewards_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"]
            )  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "val_temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                "agent_rollout": True, 
            }

            envs = self.init_envs(test_batch)
            agents = [self.agent_class() for _ in range(len(envs))]

            test_traj_generator = self.generate_agent_trajectories_async(
                envs, agents, meta_info=test_batch.meta_info
            )

            trajectories = []
            for traj in test_traj_generator: 
                trajectories.append(traj)
            test_output_gen_batch = self._transform_agent_trajectories( trajectories=trajectories)

            if test_batch.meta_info["recompute_log_prob"]:
                with torch.no_grad():
                    output = self.actor_wg.compute_log_prob(test_output_gen_batch)
                    test_output_gen_batch = test_output_gen_batch.union(output)

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
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            if data_source not in data_source_env_reward:
                data_source_env_reward[data_source] = []
            data_source_env_reward[data_source].append(env_reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        for data_source, env_rewards in data_source_env_reward.items():
            metric_dict[f"val/env_score/{data_source}"] = np.mean(env_rewards)

        return metric_dict



    def generate_agent_trajectories_async(self, envs, agents, timing_raw={}, meta_info=None):
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.interact_environment_generator(timing_raw=timing_raw, mode="Token", meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done
            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()

        while True:
            item = queue.get()
            if item is None:
                break
            yield item