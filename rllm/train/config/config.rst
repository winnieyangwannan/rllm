Config:
------
This document describes the key configuration fields used for the PPO trainer and its async/partial rollout variants.

``data.max_prompt_length``
    Maximum input prompt length in tokens. Used during tokenization, padding, and model input preparation.

``data.max_response_length``
    Maximum response length in tokens. Used to pad responses and compute attention masks.

``data.train_batch_size``
    Training batch size. Used to slice and group mini-batches across PPO update steps. Use this to manage memory pressure.

``actor_rollout_ref.model.path``
    Path to actor model (e.g., HF or local directory). Used for both rollout and inference engine.

``actor_rollout_ref.rollout.n``
    Number of rollouts per input sample. Controls sampling during both training and validation. Data used during training/validation can explore `n` times and will increase memory requirements by potentially `n` times.

``actor_rollout_ref.rollout.mode``
    ``sync`` or ``async``. Selects between synchronous and asynchronous rollout engine. Currently applicable only to AgentPPOTrainer.

``agent.max_steps``
    Set this to specify the maximum number of steps per agent trajectory in the environment.

``agent.use_stepwise_advantage``
    Enables per-step reward and advantage computation. Affects padding and broadcast logic.

``agent.stepwise_advantage_mode``
    Strategy for assigning advantage across steps. Supported: ``broadcast``, ``mc_return``.

``agent.n_parallel_agents``
    Used to control degree of parallel rollout agents in async rollout engine.

``agent.trajectory_timeout``
    Optional timeout for agent rollout execution.

``agent.normalize_step_advantage``
    Normalize advantage values across number of steps in trajectory. Used during advantage broadcast.

``agent.engine_args``
    Dictionary of engine-level args for rollout executor (e.g., vLLM async server flags).

``agent.agent_args``
    Dict of kwargs passed to agent class constructor during rollout execution.

``env.env_args``
    Dict of kwargs passed to env class constructor during rollout execution.

``algorithm.adv_estimator``
    One of ``gae``, ``grpo``, etc. Specifies advantage estimation strategy.

``algorithm.gamma``
    Discount factor for return computation.

``algorithm.lam``
    Lambda value for GAE computation.

``algorithm.mask_truncated_samples``
    Whether to zero gradients for truncated trajectories.

``algorithm.clip_advantages``
    Whether to clamp advantages to stabilize updates.

``trainer.project_name``
    Project name for logger backend (e.g., wandb, swanlab).

``trainer.experiment_name``
    Experiment name for tracking/logging.

``trainer.logger``
    List of logging backends (e.g., ``['console', 'wandb']``).

``trainer.total_epochs``
    Number of epochs to train.

``trainer.test_freq``
    Interval (in steps) to run validation.

``trainer.save_freq``
    Interval (in steps) to save checkpoints.

``trainer.critic_warmup``
    Step after which critic updates start.

``trainer.rejection_sample``
    Whether to apply rejection sampling based on rewards (filter all pass/fail samples).

``trainer.resume_mode``
    Resume mode: ``auto``, ``disable``, or ``resume_path``.

``trainer.val_before_train``
    Whether to run validation before training starts.

``trainer.total_training_steps``
    Total steps before training stops.

``reward_model.enable``
    Whether to use the reward model for scoring.

``critic.ppo_max_token_len_per_gpu``
    Used to control memory and batch shaping for critic updates.

``critic.forward_max_token_len_per_gpu``
    Same as above but applied in forward-only passes.

``critic.use_dynamic_bsz``
    Enables dynamic batch sizing for critic.

``critic.ppo_epochs``
    Number of critic update epochs.

``critic.shuffle``
    Whether to shuffle critic training batches.

``critic.grad_clip``
    Max gradient norm for critic updates.

``critic.loss_agg_mode``
    Aggregation strategy for loss computation.
