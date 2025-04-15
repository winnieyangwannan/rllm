set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 -m verl.trainer.main_ppo_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/rllm-frozenlake/train.parquet \
    data.val_files=$HOME/data/rllm-frozenlake/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=23000 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/data/colin/code/rllm/tests/rllm/agent/frozenlake/sft_model_output_qwen2.5-1.5b-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.async_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.002 \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='7b-ppo-frozenlake_agent' \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=80 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    env.name=frozenlake \
    agent.name=frozenlakeagent \
    agent.max_trajectory_length=5000 \
    agent.max_episodes=20 > output_grpo.log 2>&1